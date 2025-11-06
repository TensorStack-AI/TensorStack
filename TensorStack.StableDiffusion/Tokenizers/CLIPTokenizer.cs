//// Copyright (c) TensorStack. All rights reserved.
//// Licensed under the Apache 2.0 License.
//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Threading.Tasks;
//using TensorStack.Common;
//using TensorStack.StableDiffusion.Common;
//using TensorStack.StableDiffusion.Config;

//namespace TensorStack.StableDiffusion.Tokenizers
//{
//    public sealed class CLIPTokenizer : IDisposable
//    {
//        private readonly int _bos = 49406;
//        private readonly int _eos = 49407;
//        private readonly TokenizerConfig _configuration;
//        private readonly Microsoft.ML.Tokenizers.BpeTokenizer _tokenizer;

//        /// <summary>
//        /// Initializes a new instance of the <see cref="CLIPTokenizer"/> class.
//        /// </summary>
//        /// <param name="configuration">The configuration.</param>
//        public CLIPTokenizer(TokenizerConfig configuration)
//        {
//            _configuration = configuration;
//            _tokenizer = CreateTokenizer();
//        }

//        /// <summary>
//        /// Gets the BOS token.
//        /// </summary>
//        public long BOS => _bos;

//        /// <summary>
//        /// Gets the EOS token.
//        /// </summary>
//        public long EOS => _eos;


//        /// <summary>
//        /// Encodes the text to tokens.
//        /// </summary>
//        /// <param name="text">The text.</param>
//        /// <param name="includeBOSAndEOSTokens">if set to <c>true</c> [include bos and eos tokens].</param>
//        public Task<TokenizerResult> EncodeAsync(string text, bool includeBOSAndEOSTokens = true)
//        {
//            var resultTokensIds = new List<long>();
//            var tokensIds = _tokenizer
//                .EncodeToIds(text)
//                .ToArray()
//                .ToLong();

//            // Add BOS
//            if (includeBOSAndEOSTokens)
//                resultTokensIds.Add(_bos);

//            // Add Tokens
//            resultTokensIds.AddRange(tokensIds);

//            // AD EOS
//            if (includeBOSAndEOSTokens)
//                resultTokensIds.Add(_eos);

//            var attentionMask = Enumerable.Repeat<long>(1, resultTokensIds.Count);
//            return Task.FromResult(new TokenizerResult(resultTokensIds.ToArray(), attentionMask.ToArray()));
//        }


//        /// <summary>
//        /// Decodes the tokens to text.
//        /// </summary>
//        /// <param name="tokens">The tokens.</param>
//        public Task<string> DecodeAsync(long[] tokens)
//        {
//            var result = _tokenizer.Decode(tokens.ToInt(), false);
//            return Task.FromResult(result);
//        }


//        /// <summary>
//        /// Creates the tokenizer.
//        /// </summary>
//        /// <returns>Microsoft.ML.Tokenizers.BpeTokenizer.</returns>
//        private Microsoft.ML.Tokenizers.BpeTokenizer CreateTokenizer()
//        {
//            var directory = Path.GetDirectoryName(_configuration.Path);
//            var vocabFile = Path.Combine(directory, "vocab.json");
//            var mergesFile = Path.Combine(directory, "merges.txt");
//            return Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocabFile, mergesFile, normalizer: new Microsoft.ML.Tokenizers.LowerCaseNormalizer(), unknownToken: "<|endoftext|>", endOfWordSuffix: "</w>");
//        }


//        /// <summary>
//        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
//        /// </summary>
//        public void Dispose()
//        {
//        }
//    }
//}
