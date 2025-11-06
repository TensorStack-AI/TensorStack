// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;

namespace TensorStack.TextGeneration.Tokenizers
{
    public sealed class CLIPTokenizer : ITokenizer
    {
        private readonly int _bos = 49406;
        private readonly int _eos = 49407;
        private readonly TokenizerConfig _configuration;
        private readonly Microsoft.ML.Tokenizers.BpeTokenizer _tokenizer;
        private readonly IReadOnlyDictionary<long, string> _specialTokens;

        /// <summary>
        /// Initializes a new instance of the <see cref="CLIPTokenizer"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public CLIPTokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _tokenizer = CreateTokenizer();
            _specialTokens = _tokenizer.SpecialTokens?.ToDictionary(k => (long)k.Value, v => v.Key) ?? [];
        }

        /// <summary>
        /// Gets the BOS token.
        /// </summary>
        public long BOS => _bos;

        /// <summary>
        /// Gets the EOS token.
        /// </summary>
        public long EOS => _eos;

        /// <summary>
        /// Gets the Special Tokens
        /// </summary>
        public IReadOnlyDictionary<long, string> SpecialTokens => _specialTokens;


        /// <summary>
        /// Creates the tokenizer.
        /// </summary>
        /// <returns>Microsoft.ML.Tokenizers.BpeTokenizer.</returns>
        private Microsoft.ML.Tokenizers.BpeTokenizer CreateTokenizer()
        {
            var directory = Path.GetDirectoryName(_configuration.Path);
            var vocabFile = Path.Combine(directory, "vocab.json");
            var mergesFile = Path.Combine(directory, "merges.txt");
            return Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocabFile, mergesFile, normalizer: new Microsoft.ML.Tokenizers.LowerCaseNormalizer(), unknownToken: "<|endoftext|>", endOfWordSuffix: "</w>");
        }



        public Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> input, bool includeBOSAndEOSTokens = true)
        {
            var resultTokensIds = new List<long>();
            var tokensIds = _tokenizer
                .EncodeToIds(input)
                .ToArray()
                .ToLong();

            // Add BOS
            if (includeBOSAndEOSTokens)
                resultTokensIds.Add(_bos);

            // Add Tokens
            resultTokensIds.AddRange(tokensIds);

            // Add EOS
            if (includeBOSAndEOSTokens)
                resultTokensIds.Add(_eos);

            var attentionMask = Enumerable.Repeat<long>(1, resultTokensIds.Count);
            return Task.FromResult(new TokenizerResult(resultTokensIds.ToArray(), attentionMask.ToArray()));
        }


        public Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> input)
        {
            return EncodeAsync(input, true);
        }


        public string Decode(int token, bool considerSpecialTokens = false)
        {
            return DecodeInternal(token, considerSpecialTokens);
        }


        public string Decode(long token, bool considerSpecialTokens = false)
        {
            return DecodeInternal((int)token, considerSpecialTokens);
        }


        public string Decode(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return DecodeInternal(tokens, considerSpecialTokens);
        }


        public string Decode(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            return DecodeInternal([.. tokens.Select(Convert.ToInt32)], considerSpecialTokens);
        }


        public Task<string> DecodeAsync(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return Task.Run(() => Decode(tokens, considerSpecialTokens));
        }


        public Task<string> DecodeAsync(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            return Task.Run(() => Decode(tokens, considerSpecialTokens));
        }


        private string DecodeInternal(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return _tokenizer.Decode(tokens, considerSpecialTokens);
        }


        private string DecodeInternal(int token, bool considerSpecialTokens = false)
        {
            if (!considerSpecialTokens && _tokenizer.SpecialTokens.Values.Contains(token))
                return string.Empty;

            var value = _tokenizer.Vocabulary.FirstOrDefault(v => v.Value == token);
            return value.Key ?? string.Empty;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose() { }
    }
}
