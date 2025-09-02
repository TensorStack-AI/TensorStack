// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public class FlorenceTokenizer : ITokenizer
    {
        private readonly Regex _preTokenizeRegex;
        private readonly TokenizerConfig _configuration;
        private readonly MapCollection<byte, char> _unicodeMap;
        private readonly MapCollection<long, int> _coordinateMap;
        private readonly MapCollection<long, string> _vocabularyMap;
        private readonly MapCollection<long, string> _specialTokensMap;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlorenceTokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public FlorenceTokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _unicodeMap = CreateUnicodeMapping();
            _specialTokensMap = CreateSpecialTokenMapping();
            _vocabularyMap = CreateVocabMapping();
            _coordinateMap = CreateCoordinateMapping();
            _preTokenizeRegex = new Regex(@"'s|'t|'re|'ve|'m|'ll|'d|<loc_[\p{L}\p{N}_]+>| ?[\p{L}_][\p{L}\p{N}_]*|[^ \s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled);
        }

        public long BOS => _configuration.BOS;
        public long EOS => _configuration.EOS;
        public IReadOnlyDictionary<long, string> SpecialTokens => _specialTokensMap.AsReadOnly();


        /// <summary>
        /// Encodes the specified string to tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> text)
        {
            return EncodeAsync(text, default);
        }


        /// <summary>
        /// Encodes the specified string to tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> text, int[] coordinates)
        {
            return Task.FromResult(EncodeString(text, coordinates));
        }


        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public string Decode(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return Decode([.. tokens.Select(Convert.ToInt64)], considerSpecialTokens);
        }


        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public string Decode(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            var tokenIds = considerSpecialTokens
                ? tokens.Select(IdToToken)
                : tokens.Except(_specialTokensMap.Keys).Select(IdToToken);

            return TokensToString(tokenIds);
        }


        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public Task<string> DecodeAsync(IEnumerable<int> tokens, bool considerSpecialTokens = false)
        {
            return Task.Run(() => Decode(tokens, considerSpecialTokens));
        }


        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public Task<string> DecodeAsync(IEnumerable<long> tokens, bool considerSpecialTokens = false)
        {
            return Task.Run(() => Decode(tokens, considerSpecialTokens));
        }


        /// <summary>
        /// TokenId to Token.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public string IdToToken(long id)
        {
            if (_vocabularyMap.TryGetValue(id, out string token))
            {
                return token;
            }
            return _specialTokensMap[_configuration.UNK];
        }


        /// <summary>
        /// Token to TokenId.
        /// </summary>
        /// <param name="token">The token.</param>
        public long TokenToId(string token)
        {
            if (_vocabularyMap.TryGetValue(token, out long tokenId))
            {
                return tokenId;
            }
            return _configuration.UNK;
        }


        /// <summary>
        /// Tries the get coordinate.
        /// </summary>
        /// <param name="tokenId">The token identifier.</param>
        /// <param name="coordinate">The coordinate.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public bool TryGetCoordinate(long tokenId, out int coordinate)
        {
            return _coordinateMap.TryGetValue(tokenId, out coordinate);
        }


        /// <summary>
        /// Encodes the string.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="coordinates">The coordinates.</param>
        private TokenizerResult EncodeString(ReadOnlySpan<char> input, int[] coordinates)
        {
            var tokenized = StringToTokens(input);
            if (tokenized.Length == 0)
                return null;

            if (!coordinates.IsNullOrEmpty())
            {
                var coordinateTokens = ParseCoordinateTokens(coordinates);
                if (!coordinates.IsNullOrEmpty())
                {
                    tokenized = [.. tokenized[..^1], .. coordinateTokens, BOS];
                }
            }

            var sequenceLength = Math.Min(_configuration.MaxLength, tokenized.Length);
            var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(_configuration.MaxLength, tokenized.Length));

            var inputIds = tokenized
                .Take(_configuration.MaxLength)
                .Select(token => token)
                .Concat(padding)
                .ToArray();

            var inputMask = tokenized
                .Take(_configuration.MaxLength)
                .Select(o => 1L)
                .Concat(padding)
                .ToArray();

            return new TokenizerResult(inputIds, inputMask);
        }


        /// <summary>
        /// Tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <returns>System.String.</returns>
        private string TokensToString(IEnumerable<string> tokens)
        {
            byte[] byteArray = tokens
                .SelectMany(c => c)
                .Select(c => _unicodeMap[c])
                .ToArray();
            return Encoding.UTF8.GetString(byteArray);
        }


        /// <summary>
        /// String to TokenIds.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.Int64[].</returns>
        private long[] StringToTokens(ReadOnlySpan<char> input)
        {
            var tokens = PreTokenize(input)
                .Select(TokenToId)
                .Prepend(_configuration.BOS)
                .Append(_configuration.EOS)
                .ToArray();
            return tokens;
        }


        /// <summary>
        /// Pre-tokenize.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.String[].</returns>
        private string[] PreTokenize(ReadOnlySpan<char> input)
        {
            var tokens = _preTokenizeRegex.Matches(input.ToString())
                .Select(m => m.Value)
                .Select(t => new string
                (
                    Encoding.UTF8.GetBytes(t)
                        .Select(b => _unicodeMap[b])
                        .ToArray()
                )).ToArray();
            return tokens;
        }


        /// <summary>
        /// Parses the coordinate.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.Int32.</returns>
        /// <exception cref="System.Exception">Failed to parse {input} token</exception>
        private static int ParseCoordinate(string input)
        {
            if (!int.TryParse(input.Replace("<loc_", "").Replace(">", ""), out int position))
                throw new Exception($"Failed to parse {input} token");

            return position;
        }


        /// <summary>
        /// Parses the coordinate tokens.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns>System.Int64[].</returns>
        private long[] ParseCoordinateTokens(int[] coordinates)
        {
            var coordinateTokens = new long[coordinates.Length];
            for (int i = 0; i < coordinates.Length; i++)
            {
                int coordinate = coordinates[i];
                if (!_coordinateMap.TryGetValue(coordinate, out long tokenId))
                    return [];

                coordinateTokens[i] = tokenId;
            }
            return coordinateTokens;
        }


        /// <summary>
        /// Creates the byte to unicode mapping.
        /// </summary>
        private static MapCollection<byte, char> CreateUnicodeMapping()
        {
            var byteToUnicodeMapping = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (byte)b, b => (char)b);
            var index = 0;
            int numChars = byte.MaxValue + 1;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (!byteToUnicodeMapping.ContainsKey((byte)b))
                {
                    byteToUnicodeMapping.Add((byte)b, (char)(numChars + index));
                    ++index;
                }
            }
            return new MapCollection<byte, char>(byteToUnicodeMapping);
        }


        /// <summary>
        /// Creates the special token mapping.
        /// </summary>
        private MapCollection<long, string> CreateSpecialTokenMapping()
        {
            var tokenizerFile = Path.Combine(_configuration.Path, "tokenizer_config.json");
            using (var tokenizerReader = File.OpenRead(tokenizerFile))
            {
                var specialTokenMap = new MapCollection<long, string>();
                var config = JsonSerializer.Deserialize<TokenizerJson>(tokenizerReader);
                foreach (var addedToken in config.AddedTokens)
                {
                    specialTokenMap.TryAdd(long.Parse(addedToken.Key), addedToken.Value.Content);
                }
                return specialTokenMap;
            }
        }


        /// <summary>
        /// Creates the vocab mapping.
        /// </summary>
        private MapCollection<long, string> CreateVocabMapping()
        {
            var vocabFile = Path.Combine(_configuration.Path, "vocab.json");
            using (var vocabReader = File.OpenRead(vocabFile))
            {
                var vocab = JsonSerializer.Deserialize<Dictionary<string, long>>(vocabReader);
                var vocabularyMap = new MapCollection<long, string>(vocab);
                foreach (var addedToken in _specialTokensMap)
                {
                    vocabularyMap.TryAdd(addedToken.Key, addedToken.Value);
                }
                return vocabularyMap;
            }
        }


        /// <summary>
        /// Creates the coordinate mapping.
        /// </summary>
        private MapCollection<long, int> CreateCoordinateMapping()
        {
            var coordinateMap = new MapCollection<long, int>();
            foreach (var token in _specialTokensMap)
            {
                if (token.Value.StartsWith("<loc_"))
                    coordinateMap.Add(token.Key, ParseCoordinate(token.Value));
            }
            return coordinateMap;
        }


        public void Dispose()
        {
            _unicodeMap.Clear();
            _coordinateMap.Clear();
            _vocabularyMap.Clear();
            _specialTokensMap.Clear();
        }


        private record TokenizerJson
        {

            [JsonPropertyName("added_tokens_decoder")]
            public Dictionary<string, AddedTokenJson> AddedTokens { get; set; }
        }


        private record AddedTokenJson
        {
            [JsonPropertyName("content")]
            public string Content { get; set; }
        }
    }
}