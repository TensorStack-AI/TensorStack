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
        private readonly int[] _beginSuppressTokens = [220, 50257];
        private readonly int[] _suppressTokens = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362];


        /// <summary>
        /// Initializes a new instance of the <see cref="WhisperTokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public WhisperTokenizer(TokenizerConfig configuration)
            : base(configuration) { }

        public long NoCaptionsToken => 50362;
        public long NoTimestampToken => 50363;
        public int[] SuppressTokens => _suppressTokens;
        public int[] BeginSuppressTokens => _beginSuppressTokens;

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