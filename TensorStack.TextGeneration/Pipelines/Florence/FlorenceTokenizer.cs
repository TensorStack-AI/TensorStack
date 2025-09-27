// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public sealed partial class FlorenceTokenizer : BPETokenizer
    {
        private readonly MapCollection<long, int> _coordinateMap;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlorenceTokenizer"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public FlorenceTokenizer(TokenizerConfig configuration) 
            : base(configuration)
        {
            _coordinateMap = CreateCoordinateMapping();
        }


        /// <summary>
        /// Encodes the specified string to tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        public override Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> text)
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
        /// Creates the pre-tokenize regex.
        /// </summary>
        /// <returns>Regex.</returns>
        protected override Regex CreatePreTokenizeRegex()
        {
            return FlorencePreTokenizeRegex();
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
        /// Creates the coordinate mapping.
        /// </summary>
        private MapCollection<long, int> CreateCoordinateMapping()
        {
            var coordinateMap = new MapCollection<long, int>();
            foreach (var token in SpecialTokensMap)
            {
                if (token.Value.StartsWith("<loc_"))
                    coordinateMap.Add(token.Key, ParseCoordinate(token.Value));
            }
            return coordinateMap;
        }


        public override void Dispose()
        {
            _coordinateMap.Clear();
            base.Dispose();
        }

        [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d|<loc_[\p{L}\p{N}_]+>| ?[\p{L}_][\p{L}\p{N}_]*|[^ \s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
        private static partial Regex FlorencePreTokenizeRegex();
    }
}