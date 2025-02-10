// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Florence.Common;

namespace TensorStack.Florence.Tokenizer
{
    public class FlorenceTokenizer
    {
        private readonly FlorenceTokenizerConfig _configuration;
        private readonly Dictionary<char, byte> _unicodeToByte;
        private readonly Dictionary<byte, char> _byteToUnicode;
        private readonly Dictionary<string, long> _addedTokens;
        private readonly Dictionary<long, int> _coordinateTokens;
        private readonly Dictionary<int, long> _coordinateTokensIds;
        private readonly Dictionary<string, long> _vocabularyTokens;
        private readonly Dictionary<long, string> _vocabularyTokenIds;
        private readonly Regex _preTokenizeRegex = new Regex(@"<[^>]+>|\s*\w+|\s+|[^\w\s<>]+");

        /// <summary>
        /// Initializes a new instance of the <see cref="FlorenceTokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public FlorenceTokenizer(ModelConfig config)
        {
            _configuration = LoadConfiguration(config.Path);
            _vocabularyTokens = LoadVocabulary(config.Path);
            _vocabularyTokenIds = _vocabularyTokens.ToDictionary(k => k.Value, v => v.Key);
            _byteToUnicode = CreateByteToUnicodeMapping();
            _unicodeToByte = _byteToUnicode.ToDictionary(kv => kv.Value, kv => kv.Key);
            _coordinateTokens = CreateCoordinateTokens();
            _coordinateTokensIds = _coordinateTokens.ToDictionary(k => k.Value, v => v.Key);
            _addedTokens = _configuration.AddedTokensDecoder.ToDictionary(kv => kv.Value.Content, v => long.Parse(v.Key));
        }

        /// <summary>
        /// Gets the end of sequence identifier.
        /// </summary>
        public long EndOfSequenceId => TokenToId(_configuration.EosToken);

        /// <summary>
        /// Gets the beginning of sequence identifier.
        /// </summary>
        public long BeginningOfSequenceId => TokenToId(_configuration.BosToken);


        /// <summary>
        /// Encodes the specified prompt.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="coordinates">The coordinates.</param>
        /// <returns>TokenizerResult.</returns>
        public TokenizerResult Encode(string prompt, int[] coordinates = null)
        {
            var tokenized = StringToTokens(prompt);
            if (tokenized.Length == 0)
                return null;

            if (!coordinates.IsNullOrEmpty())
            {
                var coordinateTokens = ParseCoordinateTokens(coordinates);
                if (!coordinates.IsNullOrEmpty())
                {
                    tokenized = [.. tokenized[..^1], .. coordinateTokens, EndOfSequenceId];
                }
            }

            var sequenceLength = Math.Min(_configuration.ModelMaxLength, tokenized.Length);
            var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(_configuration.ModelMaxLength, tokenized.Length));

            var inputIds = tokenized
                .Take(_configuration.ModelMaxLength)
                .Select(token => token)
                .Concat(padding)
                .ToArray();

            var inputMask = tokenized
                .Take(_configuration.ModelMaxLength)
                .Select(o => 1L)
                .Concat(padding)
                .ToArray();

            int[] dimensions = [1, inputIds.Length];
            return new TokenizerResult(new Tensor<long>(inputIds, dimensions), new Tensor<long>(inputMask, dimensions));
        }


        /// <summary>
        /// Decodes the specified token ids.
        /// </summary>
        /// <param name="tokenIds">The token ids.</param>
        /// <param name="specialTokens">if set to <c>true</c> [special tokens].</param>
        /// <returns>System.String.</returns>
        public string Decode(IReadOnlyCollection<long> tokenIds, bool specialTokens = true)
        {
            var tokens = specialTokens
                ? tokenIds.Select(IdToToken)
                : tokenIds.Except(_addedTokens.Values).Select(IdToToken);
            var decoded = TokensToString(tokens);
            return SanitizeInput(decoded);
        }


        /// <summary>
        /// Identifiers to token.
        /// </summary>
        /// <param name="id">The identifier.</param>
        /// <returns>System.String.</returns>
        public string IdToToken(long id)
        {
            if (_vocabularyTokenIds.TryGetValue((int)id, out string token))
            {
                return token;
            }
            return _configuration.UnkToken;
        }


        /// <summary>
        /// Tokens to identifier.
        /// </summary>
        /// <param name="token">The token.</param>
        /// <returns>System.Int64.</returns>
        public long TokenToId(string token)
        {
            if (_vocabularyTokens.TryGetValue(token, out long tokenId))
            {
                return tokenId;
            }
            return _vocabularyTokens[_configuration.UnkToken];
        }


        /// <summary>
        /// Tries the get coordinate.
        /// </summary>
        /// <param name="tokenId">The token identifier.</param>
        /// <param name="coordinate">The coordinate.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public bool TryGetCoordinate(long tokenId, out int coordinate)
        {
            return _coordinateTokens.TryGetValue(tokenId, out coordinate);
        }


        /// <summary>
        /// Loads the configuration.
        /// </summary>
        /// <param name="tokenizerPath">The tokenizer path.</param>
        /// <returns>FlorenceTokenizerConfig.</returns>
        /// <exception cref="System.IO.FileNotFoundException">tokenizer_config.json not found</exception>
        private static FlorenceTokenizerConfig LoadConfiguration(string tokenizerPath)
        {
            var configPath = Path.Combine(tokenizerPath, "tokenizer_config.json");
            if (!File.Exists(configPath))
                throw new FileNotFoundException("tokenizer_config.json not found");

            var fileTokenizerConfigJson = File.ReadAllText(configPath);
            return JsonSerializer.Deserialize<FlorenceTokenizerConfig>(fileTokenizerConfigJson);
        }


        /// <summary>
        /// Loads the vocabulary.
        /// </summary>
        /// <param name="tokenizerPath">The tokenizer path.</param>
        /// <returns>Dictionary&lt;System.String, System.Int64&gt;.</returns>
        /// <exception cref="System.IO.FileNotFoundException">vocab.json not found</exception>
        private Dictionary<string, long> LoadVocabulary(string tokenizerPath)
        {
            var fileVocabPath = Path.Combine(tokenizerPath, "vocab.json");
            if (!File.Exists(fileVocabPath))
                throw new FileNotFoundException("vocab.json not found");

            var vocabJson = File.ReadAllText(fileVocabPath);
            var vocabulary = JsonSerializer.Deserialize<Dictionary<string, long>>(vocabJson);
            foreach (var addedToken in _configuration.AddedTokensDecoder)
            {
                if (!vocabulary.TryGetValue(addedToken.Value.Content, out long _))
                {
                    vocabulary.Add(addedToken.Value.Content, long.Parse(addedToken.Key));
                }
            }
            return vocabulary;
        }


        /// <summary>
        /// Tokenses to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <returns>System.String.</returns>
        private string TokensToString(IEnumerable<string> tokens)
        {
            byte[] byteArray = tokens
                .SelectMany(c => c)
                .Select(c => _unicodeToByte[c])
                .ToArray();
            return Encoding.UTF8.GetString(byteArray);
        }


        /// <summary>
        /// Strings to tokens.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>System.Int64[].</returns>
        private long[] StringToTokens(string text)
        {
            var tokens = PreTokenize(text)
                .Prepend(_configuration.ClsToken)
                .Append(_configuration.SepToken)
                .Select(TokenToId)
                .ToArray();
            return tokens;
        }


        /// <summary>
        /// Pres the tokenize.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>System.String[].</returns>
        private string[] PreTokenize(string text)
        {
            var tokens = _preTokenizeRegex.Matches(text)
                .Select(m => m.Value)
                .Select(t => new string
                (
                    Encoding.UTF8.GetBytes(t)
                        .Select(b => _byteToUnicode[b])
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
                if (!_coordinateTokensIds.TryGetValue(coordinate, out long tokenId))
                    return [];

                coordinateTokens[i] = tokenId;
            }
            return coordinateTokens;
        }


        /// <summary>
        /// Creates the coordinate tokens.
        /// </summary>
        /// <returns>Dictionary&lt;System.Int64, System.Int32&gt;.</returns>
        private Dictionary<long, int> CreateCoordinateTokens()
        {
            return _vocabularyTokens
                .Where(x => x.Key.StartsWith("<loc_"))
                .ToDictionary(k => k.Value, v => ParseCoordinate(v.Key));
        }


        /// <summary>
        /// Sanitizes the input (spaces before punctuations and abbreviated forms)
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>System.String.</returns>
        private static string SanitizeInput(string text)
        {
            return text.Replace(" .", ".")
               .Replace(" ?", "?")
               .Replace(" !", "!")
               .Replace(" ,", ",")
               .Replace(" ' ", "")
               .Replace(" n't", "n't")
               .Replace(" 'm", "'m")
               .Replace(" 's", "'s")
               .Replace(" 've", "'ve")
               .Replace(" 're", "'re");
        }


        /// <summary>
        /// Creates the byte to unicode mapping.
        /// </summary>
        /// <returns>Dictionary&lt;System.Byte, System.Char&gt;.</returns>
        private static Dictionary<byte, char> CreateByteToUnicodeMapping()
        {
            var byteToUnicodeMapping = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (byte)b, b => (char)b);
            var index = 0;
            int numChars = 256;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (!byteToUnicodeMapping.ContainsKey((byte)b))
                {
                    byteToUnicodeMapping.Add((byte)b, (char)(numChars + index));
                    ++index;
                }
            }
            return byteToUnicodeMapping;
        }
    }

}