//// Copyright (c) TensorStack. All rights reserved.
//// Licensed under the Apache 2.0 License.
//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Text.Json;
//using System.Text.Json.Serialization;
//using System.Threading.Tasks;
//using TensorStack.Common;
//using TensorStack.StableDiffusion.Common;
//using TensorStack.StableDiffusion.Config;

//namespace TensorStack.StableDiffusion.Tokenizers
//{
//    public sealed class T5Tokenizer : IDisposable
//    {
//        private readonly int _bos = 0;
//        private readonly int _eos = 1;
//        private readonly TokenizerConfig _configuration;
//        private readonly Microsoft.ML.Tokenizers.SentencePieceTokenizer _tokenizer;

//        /// <summary>
//        /// Initializes a new instance of the <see cref="T5Tokenizer"/> class.
//        /// </summary>
//        /// <param name="configuration">The configuration.</param>
//        public T5Tokenizer(TokenizerConfig configuration)
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
//        public Task<TokenizerResult> EncodeAsync(string text)
//        {
//            var inputIds = _tokenizer.EncodeToIds(text).ToArray().ToLong();
//            var attentionMask = Enumerable.Repeat<long>(1, inputIds.Length).ToArray();
//            return Task.FromResult(new TokenizerResult(inputIds, attentionMask));
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
//        /// <returns>Microsoft.ML.Tokenizers.SentencePieceTokenizer.</returns>
//        private Microsoft.ML.Tokenizers.SentencePieceTokenizer CreateTokenizer()
//        {
//            var specialTokens = GetSpecialTokens(_configuration.Path);
//            using (var fileStream = File.OpenRead(_configuration.Path))
//            {
//                return Microsoft.ML.Tokenizers.SentencePieceTokenizer.Create(fileStream, addBeginOfSentence: false, addEndOfSentence: true, specialTokens);
//            }
//        }


//        /// <summary>
//        /// Gets the special tokens.
//        /// </summary>
//        /// <param name="tokeizerModelPath">The tokeizer model path.</param>
//        private Dictionary<string, int> GetSpecialTokens(string tokeizerModelPath)
//        {
//            try
//            {
//                var tokenizerConfig = Path.Combine(Path.GetDirectoryName(tokeizerModelPath), "tokenizer.json");
//                if (!File.Exists(tokenizerConfig))
//                    return null;

//                using (var tokenizerConfigFile = File.OpenRead(tokenizerConfig))
//                {
//                    var sentencePieceConfig = JsonSerializer.Deserialize<SentencePieceConfig>(tokenizerConfigFile);
//                    if (sentencePieceConfig is null || sentencePieceConfig.AddedTokens is null)
//                        return null;


//                    return sentencePieceConfig.AddedTokens.ToDictionary(k => k.Content, v => v.Id);
//                }
//            }
//            catch (Exception)
//            {
//                return null;
//            }
//        }


//        /// <summary>
//        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
//        /// </summary>
//        public void Dispose()
//        {
//        }


//        private record SentencePieceConfig
//        {
//            [JsonPropertyName("added_tokens")]
//            public AddedToken[] AddedTokens { get; set; }
//        }


//        private record AddedToken
//        {
//            [JsonPropertyName("id")]
//            public int Id { get; set; }

//            [JsonPropertyName("content")]
//            public string Content { get; set; }
//        }
//    }
//}
