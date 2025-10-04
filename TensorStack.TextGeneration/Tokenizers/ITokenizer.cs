// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorStack.TextGeneration.Tokenizers
{
    public interface ITokenizer : IDisposable
    {
        /// <summary>
        /// Gets the Begin Of Sentence token.
        /// </summary>
        long BOS { get; }

        /// <summary>
        /// Gets the End Of Sentence token.
        /// </summary>
        long EOS { get; }

        /// <summary>
        /// Gets the Special Tokens
        /// </summary>
        IReadOnlyDictionary<long, string> SpecialTokens { get; }

        /// <summary>
        /// Encodes the specified string to tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        Task<TokenizerResult> EncodeAsync(ReadOnlySpan<char> input);

        /// <summary>
        /// Decodes the specified token.
        /// </summary>
        /// <param name="token">The token.</param>
        string Decode(int token, bool considerSpecialTokens = false);

        /// <summary>
        /// Decodes the specified token.
        /// </summary>
        /// <param name="token">The token.</param>
        string Decode(long token, bool considerSpecialTokens = false);

        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        string Decode(IEnumerable<int> tokens, bool considerSpecialTokens = false);

        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        string Decode(IEnumerable<long> tokens, bool considerSpecialTokens = false);

        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        Task<string> DecodeAsync(IEnumerable<int> tokens, bool considerSpecialTokens = false);

        /// <summary>
        /// Decodes the specified tokens to string.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="considerSpecialTokens">if set to <c>true</c> decode special tokens.</param>
        Task<string> DecodeAsync(IEnumerable<long> tokens, bool considerSpecialTokens = false);

      
    }
}