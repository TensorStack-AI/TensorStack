// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Tokenizers;

namespace TensorStack.StableDiffusion.Helpers
{
    public static class PromptParser
    {
        private const float EmphasisMultiplier = 1.1f;
        private const float DeemphasisMultiplier = 0.9f;

        /// <summary>
        /// Tokenize the prompt prompt and parse the token weights.
        /// </summary>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="promptText">The text.</param>
        public static async Task<TokenizerResult> TokenizePromptAsync(CLIPTokenizer tokenizer, string promptText)
        {
            if (string.IsNullOrEmpty(promptText))
                return new TokenizerResult([], [], []);

            var tokenIds = new List<long> { tokenizer.BOS }; // bos
            var tokenWeights = new List<float> { 1 }; // bos
            var fragments = Parse(promptText);
            foreach (var fragment in fragments)
            {
                var fragmentTokens = await tokenizer.EncodeAsync(fragment.Text, false);
                tokenIds.AddRange(fragmentTokens.InputIds);
                tokenWeights.AddRange(Enumerable.Repeat(fragment.Weight, fragmentTokens.InputIds.Length));
            }

            tokenIds.Add(tokenizer.EOS); //eos
            tokenWeights.Add(1); //eos
            var attentionMask = Enumerable.Repeat<long>(1, tokenIds.Count);
            return new TokenizerResult(tokenIds.ToArray(), attentionMask.ToArray(), tokenWeights.ToArray());
        }


        /// <summary>
        /// Encode prompt batching if required.
        /// </summary>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <param name="hiddenStateIndex">Index of the hidden state.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A Task&lt;TextEncoderResult&gt; representing the asynchronous operation.</returns>
        public static async Task<TextEncoderResult> EncodePromptAsync(CLIPTextModel textEncoder, TokenizerResult inputTokens, int minimumLength, int hiddenStateIndex = 0, CancellationToken cancellationToken = default)
        {
            if (minimumLength <= textEncoder.SequenceLength)
            {
                var textEncoderResult = await textEncoder.RunAsync(inputTokens, cancellationToken);
                ApplyPromptWeights(inputTokens, textEncoderResult);
                return new TextEncoderResult(textEncoderResult.GetHiddenStates(hiddenStateIndex), textEncoderResult.TextEmbeds);
            }
            else
            {
                var bos = inputTokens.InputIds[0];
                var eos = inputTokens.InputIds[^1];
                var tokenIds = inputTokens.InputIds[1..^1].Pad(textEncoder.PadTokenId, minimumLength);

                // Create batches, 75 tokens + EOS & BOS
                var chunkSize = textEncoder.SequenceLength - 2;
                var tokenResults = new List<long[]>();
                var attentionChunks = new List<long[]>();
                var tokenChunks = tokenIds.Chunk(chunkSize).ToArray();
                for (int i = 0; i < tokenChunks.Length; i++)
                {
                    var tokenChunk = tokenChunks[i];
                    var attentionChunk = new long[textEncoder.SequenceLength];
                    var padIndex = tokenChunk.IndexOf(textEncoder.PadTokenId);
                    if ((i == 0 && tokenChunks.Length > 1) || (i > 0 && padIndex < 0))
                    {
                        tokenChunk = [.. tokenChunk.Prepend(bos), eos];
                        attentionChunk = [.. Enumerable.Repeat<long>(1, tokenChunk.Length)];
                    }
                    else
                    {
                        tokenChunk[padIndex] = eos;
                        tokenChunk = [.. tokenChunk.Prepend(bos)];
                        for (int a = 0; a < padIndex + 2; a++)
                            attentionChunk[a] = 1;

                        tokenChunk = tokenChunk.Pad(textEncoder.PadTokenId, textEncoder.SequenceLength);
                        attentionChunk = attentionChunk.Pad(0, textEncoder.SequenceLength);
                    }
                    tokenResults.Add(tokenChunk);
                    attentionChunks.Add(attentionChunk);
                }


                // Compile token batches removing duplicated EOS & BOS embeddings
                var promptEmbeds = new List<float>();
                var promptPooledEmbeds = new List<float>();
                for (int i = 0; i < tokenResults.Count; i++)
                {
                    var result = await textEncoder.RunAsync(new TokenizerResult(tokenResults[i], attentionChunks[i]), cancellationToken);
                    var indexedHiddenStates = result.GetHiddenStates(hiddenStateIndex);
                    if (i == 0 && tokenResults.Count > 1)
                    {
                        // First chunk remove EOS token
                        var output = new float[(indexedHiddenStates.Dimensions[1] - 1) * indexedHiddenStates.Dimensions[2]];
                        indexedHiddenStates.Span[..output.Length].CopyTo(output);
                        promptEmbeds.AddRange(output);
                    }
                    else if (i > 0 && i != (tokenResults.Count - 1))
                    {
                        // Middle chunk remove EOS and BOS tokens
                        var output = new float[(indexedHiddenStates.Dimensions[1] - 2) * indexedHiddenStates.Dimensions[2]];
                        indexedHiddenStates.Span.Slice(indexedHiddenStates.Dimensions[2], output.Length).CopyTo(output);
                        promptEmbeds.AddRange(output);
                    }
                    else
                    {
                        // Last chunk remove BOS token
                        var output = new float[(indexedHiddenStates.Dimensions[1] - 1) * indexedHiddenStates.Dimensions[2]];
                        indexedHiddenStates.Span.Slice(indexedHiddenStates.Dimensions[2], output.Length).CopyTo(output);
                        promptEmbeds.AddRange(output);
                    }

                    promptPooledEmbeds.AddRange(result.TextEmbeds.Span);
                }

                var hiddenStates = new Tensor<float>(promptEmbeds.ToArray(), [1, promptEmbeds.Count / textEncoder.HiddenSize, textEncoder.HiddenSize]);
                var textEmbeds = new Tensor<float>(promptPooledEmbeds.ToArray(), [1, promptPooledEmbeds.Count / textEncoder.HiddenSize, textEncoder.HiddenSize]);
                var textEncoderResult = new TextEncoderResult(hiddenStates, textEmbeds);
                ApplyPromptWeights(inputTokens, textEncoderResult);
                return textEncoderResult;
            }
        }


        /// <summary>
        /// Applies the prompt weights.
        /// </summary>
        /// <param name="tokenizerOutput">The tokenizer output.</param>
        /// <param name="encoderOutput">The encoder output.</param>
        public static void ApplyPromptWeights(TokenizerResult tokenizerOutput, TextEncoderResult encoderOutput)
        {
            var hiddenStates = encoderOutput.HiddenStates;
            var numTokens = hiddenStates.Dimensions[1];
            var embedDim = hiddenStates.Dimensions[2];
            var weights = tokenizerOutput.Weights.Pad(1, numTokens);
            if (weights.All(x => x == 1))
                return;

            var buffer = hiddenStates.Memory.Span;
            for (int tokenIndex = 0; tokenIndex < numTokens; tokenIndex++)
            {
                var weight = weights[tokenIndex];
                if (weight == 1)
                    continue;

                var offset = tokenIndex * embedDim;
                var tokenSpan = buffer.Slice(offset, embedDim);
                System.Numerics.Tensors.TensorPrimitives.Multiply(tokenSpan, weight, tokenSpan);
            }
        }


        /// <summary>
        /// Parses the specified prompt.
        /// 
        /// (text:weight) explicit weighting
        /// () implied emphasis(x EmphasisMultiplier)
        /// [] implied de-emphasis(x DeemphasisMultiplier)
        /// 
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        private static List<PromptFragment> Parse(string prompt)
        {
            var fragments = new List<PromptFragment>();
            var stack = new Stack<float>();
            float currentWeight = 1.0f;
            int i = 0;
            var buffer = new StringBuilder();

            while (i < prompt.Length)
            {
                char c = prompt[i];

                if (c == '(')
                {
                    // Flush buffer before entering new group
                    FlushBuffer(buffer, currentWeight, fragments);

                    // Check for explicit weight
                    int end = FindClosing(prompt, i, '(', ')');
                    if (end > i)
                    {
                        int colonIndex = prompt.LastIndexOf(':', end - 1, end - i);
                        if (colonIndex > i)
                        {
                            string innerText = prompt.Substring(i + 1, colonIndex - i - 1);
                            string weightText = prompt.Substring(colonIndex + 1, end - colonIndex - 1);

                            if (float.TryParse(weightText, out float weight))
                            {
                                fragments.Add(new PromptFragment(innerText.Trim(), currentWeight * weight));
                                i = end + 1;
                                continue;
                            }
                        }
                    }

                    // Implied emphasis
                    stack.Push(currentWeight);
                    currentWeight *= EmphasisMultiplier;
                    i++;
                }
                else if (c == '[')
                {
                    FlushBuffer(buffer, currentWeight, fragments);
                    stack.Push(currentWeight);
                    currentWeight *= DeemphasisMultiplier;
                    i++;
                }
                else if (c == ')' || c == ']')
                {
                    FlushBuffer(buffer, currentWeight, fragments);
                    currentWeight = stack.Count > 0 ? stack.Pop() : 1.0f;
                    i++;
                }
                else
                {
                    buffer.Append(c);
                    i++;
                }
            }

            // Final flush
            FlushBuffer(buffer, currentWeight, fragments);

            return fragments;
        }


        private static int FindClosing(string prompt, int start, char openChar, char closeChar)
        {
            int depth = 0;
            for (int i = start; i < prompt.Length; i++)
            {
                if (prompt[i] == openChar) depth++;
                else if (prompt[i] == closeChar)
                {
                    depth--;
                    if (depth == 0) return i;
                }
            }
            return -1;
        }


        private static void FlushBuffer(StringBuilder buffer, float weight, List<PromptFragment> fragments)
        {
            if (buffer.Length > 0)
            {
                string text = buffer.ToString().Trim();
                if (!string.IsNullOrWhiteSpace(text))
                    fragments.Add(new PromptFragment(text, weight));
                buffer.Clear();
            }
        }

        private record PromptFragment(string Text, float Weight = 1.0f);
    }
}
