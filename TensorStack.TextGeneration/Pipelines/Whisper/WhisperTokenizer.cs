// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorStack.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public sealed class WhisperTokenizer : BPETokenizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="WhisperTokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public WhisperTokenizer(TokenizerConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Pre-tokenize.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.String[].</returns>
        protected override string[] PreTokenize(ReadOnlySpan<char> input)
        {
            var text = input.ToString();
            var tokens = new List<string>();

            // First, extract any <|...|> special tokens
            var specials = SpecialTokensMap.Values
                .OrderByDescending(s => s.Length) // longest match first
                .ToArray();

            int idx = 0;
            while (idx < text.Length)
            {
                var match = specials.FirstOrDefault(s =>
                    idx + s.Length <= text.Length &&
                    text.AsSpan(idx, s.Length).SequenceEqual(s));

                if (match is not null)
                {
                    tokens.Add(match);
                    idx += match.Length;
                }
                else
                {
                    // Collect a single character until regex phase
                    tokens.Add(text[idx].ToString());
                    idx++;
                }
            }

            // Join non-special chunks and run regex on them
            var finalTokens = new List<string>();
            foreach (var t in tokens)
            {
                if (SpecialTokensMap.Values.Contains(t))
                {
                    finalTokens.Add(t);
                }
                else
                {
                    var regexMatches = PreTokenizeRegex.Matches(t)
                        .Select(m => m.Value)
                        .Select(str => new string(
                            Encoding.UTF8.GetBytes(str)
                                .Select(b => UnicodeMap[b])
                                .ToArray()));

                    finalTokens.AddRange(regexMatches);
                }
            }

            return finalTokens.ToArray();
        }

    }
}