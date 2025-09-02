// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using TensorStack.Common;

namespace TensorStack.TextGeneration.Tokenizers
{
    public sealed class T5Tokenizer : ITokenizer
    {
        private readonly TokenizerConfig _configuration;
        private readonly SentencePieceTokenizer _tokenizer;
        private readonly IReadOnlyDictionary<long, string> _specialTokens;

        /// <summary>
        /// Initializes a new instance of the <see cref="T5Tokenizer"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public T5Tokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _tokenizer = CreateTokenizer();
            _specialTokens = _tokenizer.SpecialTokens.ToDictionary(k => (long)k.Value, v => v.Key);
        }

        /// <summary>
        /// Gets the BOS token.
        /// </summary>
        public long BOS => _configuration.BOS;

        /// <summary>
        /// Gets the EOS token.
        /// </summary>
        public long EOS => _configuration.EOS;

        /// <summary>
        /// Gets the special tokens.
        /// </summary>
        public IReadOnlyDictionary<long, string> SpecialTokens => _specialTokens;


        /// <summary>
        /// Encodes the text to tokens.
        /// </summary>
        /// <param name="text">The text.</param>
        public Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> text)
        {
            var tokens = _tokenizer.EncodeToTokens(text, out var normalizedText, false, false);
            var inputIds = tokens.Select(x => Convert.ToInt64(x.Id)).ToArray();
            var attentionMask = Enumerable.Repeat<long>(1, inputIds.Length).ToArray();
            return Task.FromResult(new TokenizerResult(inputIds, attentionMask, normalizedText));
        }


        /// <summary>
        /// Decodes the tokens to text.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> [consider special tokens].</param>
        public string Decode(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return _tokenizer.Decode(tokens, considerSpecialTokens);
        }


        /// <summary>
        /// Decodes the tokens to text.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> [consider special tokens].</param>
        public string Decode(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            return Decode([.. tokens.Select(Convert.ToInt32)], considerSpecialTokens);
        }


        /// <summary>
        /// Decodes the tokens to text.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> [consider special tokens].</param>
        public Task<string> DecodeAsync(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return Task.FromResult(Decode(tokens, considerSpecialTokens));
        }


        /// <summary>
        /// Decodes the tokens to text.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> [consider special tokens].</param>
        public Task<string> DecodeAsync(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            return Task.FromResult(Decode([.. tokens.Select(Convert.ToInt32)], considerSpecialTokens));
        }


        /// <summary>
        /// Creates the tokenizer.
        /// </summary>
        /// <returns>SentencePieceTokenizer.</returns>
        private SentencePieceTokenizer CreateTokenizer()
        {
            var specialTokens = GetSpecialTokens(_configuration.Path);
            using (var fileStream = File.OpenRead(_configuration.Path))
            {
                return SentencePieceTokenizer.Create(fileStream, addBeginOfSentence: false, addEndOfSentence: true, specialTokens);
            }
        }


        /// <summary>
        /// Gets the special tokens.
        /// </summary>
        /// <param name="tokeizerModelPath">The tokeizer model path.</param>
        private Dictionary<string, int> GetSpecialTokens(string tokeizerModelPath)
        {
            try
            {
                var tokenizerConfig = Path.Combine(Path.GetDirectoryName(tokeizerModelPath), "tokenizer.json");
                if (!File.Exists(tokenizerConfig))
                    return null;

                using (var tokenizerConfigFile = File.OpenRead(tokenizerConfig))
                {
                    var sentencePieceConfig = JsonSerializer.Deserialize<SentencePieceConfig>(tokenizerConfigFile);
                    if (sentencePieceConfig is null || sentencePieceConfig.AddedTokens is null)
                        return null;


                    return sentencePieceConfig.AddedTokens.ToDictionary(k => k.Content, v => v.Id);
                }
            }
            catch (Exception)
            {
                return null;
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
        }


        private record SentencePieceConfig
        {
            [JsonPropertyName("added_tokens")]
            public AddedToken[] AddedTokens { get; set; }
        }


        private record AddedToken
        {
            [JsonPropertyName("id")]
            public int Id { get; set; }

            [JsonPropertyName("content")]
            public string Content { get; set; }
        }
    }
}
