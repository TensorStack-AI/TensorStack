// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.TextGeneration.Processing
{
    public class SequenceComparer : IEqualityComparer<Sequence>
    {
        private readonly HashSet<long> _specialTokens;
        private int _compareLength;

        /// <summary>
        /// Initializes a new instance of the <see cref="SequenceComparer"/> class.
        /// </summary>
        /// <param name="specialTokens">The special tokens.</param>
        /// <param name="compareLength">Length of the compare.</param>
        public SequenceComparer(IReadOnlyDictionary<long, string> specialTokens, int compareLength = int.MaxValue)
        {
            SetLength(compareLength);
            _specialTokens = [.. specialTokens.Keys];
        }


        /// <summary>
        /// Determines whether the specified objects are equal.
        /// </summary>
        /// <param name="x">The first object of type <paramref name="T" /> to compare.</param>
        /// <param name="y">The second object of type <paramref name="T" /> to compare.</param>
        /// <returns><see langword="true" /> if the specified objects are equal; otherwise, <see langword="false" />.</returns>
        public bool Equals(Sequence x, Sequence y)
        {
            if (x == null || y == null)
                return false;

            var normX = NormalizeTokens(x.Tokens);
            var normY = NormalizeTokens(y.Tokens);
            return normX.SequenceEqual(normY);
        }


        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <param name="obj">The <see cref="T:System.Object" /> for which a hash code is to be returned.</param>
        /// <returns>A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table.</returns>
        public int GetHashCode(Sequence obj)
        {
            unchecked
            {
                int hash = 17;
                foreach (var val in NormalizeTokens(obj.Tokens))
                    hash = hash * 23 + val.GetHashCode();
                return hash;
            }
        }


        /// <summary>
        /// Sets the length.
        /// </summary>
        /// <param name="length">The length.</param>
        public void SetLength(int length)
        {
            _compareLength = Math.Max(1, length);
        }


        /// <summary>
        /// Normalizes the tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <returns>IEnumerable&lt;System.Int64&gt;.</returns>
        private IEnumerable<long> NormalizeTokens(IReadOnlyList<long> tokens)
        {
            foreach (var t in tokens.Except(_specialTokens).Take(_compareLength))
            {
                yield return t;
            }
        }
    }
}