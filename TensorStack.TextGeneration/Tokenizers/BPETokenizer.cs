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

namespace TensorStack.TextGeneration.Tokenizers
{
    public partial class BPETokenizer : ITokenizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BPETokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public BPETokenizer(TokenizerConfig configuration)
        {
            Configuration = configuration;
            UnicodeMap = CreateUnicodeMapping();
            SpecialTokensMap = CreateSpecialTokenMapping();
            VocabularyMap = CreateVocabMapping();
            MergesMap = CreateMergesMapping();
            PreTokenizeRegex = CreatePreTokenizeRegex();
        }

        protected TokenizerConfig Configuration { get; }
        protected MapCollection<byte, char> UnicodeMap { get; }
        protected MapCollection<long, string> VocabularyMap { get; }
        protected Dictionary<MergeToken, int> MergesMap { get; }
        protected MapCollection<long, string> SpecialTokensMap { get; }
        protected Regex PreTokenizeRegex { get; }

        public long BOS => Configuration.BOS;
        public long EOS => Configuration.EOS;
        public IReadOnlyDictionary<long, string> SpecialTokens => SpecialTokensMap.AsReadOnly();


        /// <summary>
        /// Encodes the specified string to tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public virtual Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> text)
        {
            return Task.FromResult(EncodeString(text));
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
                : tokens.Except(SpecialTokensMap.Keys).Select(IdToToken);

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
        /// Decodes the specified token.
        /// </summary>
        /// <param name="token">The token.</param>
        public string Decode(int token, bool considerSpecialTokens = false)
        {
            return Decode(token);
        }


        /// <summary>
        /// Decodes the specified token.
        /// </summary>
        /// <param name="token">The token.</param>
        public string Decode(long token, bool considerSpecialTokens = false)
        {
            if (!considerSpecialTokens && SpecialTokensMap.ContainsKey(token))
                return string.Empty;

            return TokensToString([VocabularyMap[token]]);
        }


        /// <summary>
        /// TokenId to Token.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public string IdToToken(long id)
        {
            if (VocabularyMap.TryGetValue(id, out string token))
            {
                return token;
            }
            return VocabularyMap[Configuration.UNK];
        }


        /// <summary>
        /// Token to TokenId.
        /// </summary>
        /// <param name="token">The token.</param>
        public long TokenToId(string token)
        {
            if (VocabularyMap.TryGetValue(token, out long tokenId))
            {
                return tokenId;
            }
            return Configuration.UNK;
        }


        /// <summary>
        /// Creates the pre tokenize regex.
        /// </summary>
        /// <returns>Regex.</returns>
        protected virtual Regex CreatePreTokenizeRegex()
        {
            return DefaultPreTokenizeRegex();
        }


        /// <summary>
        /// Pre-tokenize.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.String[].</returns>
        protected virtual string[] PreTokenize(ReadOnlySpan<char> input)
        {
            var tokens = PreTokenizeRegex.Matches(input.ToString())
                .Select(m => m.Value)
                .Select(t => new string
                (
                    Encoding.UTF8.GetBytes(t)
                        .Select(b => UnicodeMap[b])
                        .ToArray()
                )).ToArray();
            return tokens;
        }


        /// <summary>
        /// Encodes the string.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="coordinates">The coordinates.</param>
        protected virtual TokenizerResult EncodeString(ReadOnlySpan<char> input)
        {
            var tokenized = StringToTokens(input);
            if (tokenized.Length == 0)
                return null;

            var sequenceLength = Math.Min(Configuration.MaxLength, tokenized.Length);
            var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(Configuration.MaxLength, tokenized.Length));

            var inputIds = tokenized
                .Take(Configuration.MaxLength)
                .Select(token => token)
                .Concat(padding)
                .ToArray();

            var inputMask = tokenized
                .Take(Configuration.MaxLength)
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
        protected string TokensToString(IEnumerable<string> tokens)
        {
            byte[] byteArray = tokens
                .SelectMany(c => c)
                .Select(c => UnicodeMap[c])
                .ToArray();
            return Encoding.UTF8.GetString(byteArray);
        }


        /// <summary>
        /// String to TokenIds.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>System.Int64[].</returns>
        protected long[] StringToTokens(ReadOnlySpan<char> input)
        {
            var tokens = PreTokenize(input)
                .SelectMany(ApplyMerges)
                .Select(TokenToId)
                .Prepend(Configuration.BOS)
                .Append(Configuration.EOS)
                .ToArray();
            return tokens;
        }


        /// <summary>
        /// Applies the merges.
        /// </summary>
        /// <param name="token">The token.</param>
        protected List<string> ApplyMerges(string token)
        {
            if (SpecialTokensMap.ContainsKey(token))
                return [token];

            var symbols = token.Select(c => c.ToString()).ToList();
            while (symbols.Count > 1)
            {
                int bestIndex = -1;
                int bestRank = int.MaxValue;
                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    var pair = new MergeToken(symbols[i], symbols[i + 1]);
                    if (MergesMap.TryGetValue(pair, out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }

                if (bestIndex == -1)
                    break;

                // Merge the pair
                string merged = symbols[bestIndex] + symbols[bestIndex + 1];
                symbols[bestIndex] = merged;
                symbols.RemoveAt(bestIndex + 1);
            }
            return symbols;
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
            var tokenizerFile = Path.Combine(Configuration.Path, "tokenizer_config.json");
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
            var vocabFile = Path.Combine(Configuration.Path, "vocab.json");
            using (var vocabReader = File.OpenRead(vocabFile))
            {
                var vocab = JsonSerializer.Deserialize<Dictionary<string, long>>(vocabReader);
                var vocabularyMap = new MapCollection<long, string>(vocab);
                foreach (var addedToken in SpecialTokensMap)
                {
                    vocabularyMap.TryAdd(addedToken.Key, addedToken.Value);
                }
                return vocabularyMap;
            }
        }


        /// <summary>
        /// Creates the merges mapping.
        /// </summary>
        private Dictionary<MergeToken, int> CreateMergesMapping()
        {
            var mergesFile = Path.Combine(Configuration.Path, "merges.txt");
            return File.ReadLines(mergesFile)
                .Where(line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith("#"))
                .Select((line, index) =>
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return (new MergeToken(parts[0], parts[1]), index);
                }).ToDictionary(x => x.Item1, x => x.index);
        }


        public virtual void Dispose()
        {
            UnicodeMap.Clear();
            VocabularyMap.Clear();
            SpecialTokensMap.Clear();
        }


        protected record TokenizerJson
        {

            [JsonPropertyName("added_tokens_decoder")]
            public Dictionary<string, AddedTokenJson> AddedTokens { get; set; }
        }


        protected record AddedTokenJson
        {
            [JsonPropertyName("content")]
            public string Content { get; set; }
        }

        protected record MergeToken(string PartA, string PartB);



        [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
        private static partial Regex DefaultPreTokenizeRegex();
    }
}

