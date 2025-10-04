// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;

namespace TensorStack.TextGeneration.Processing
{
    public sealed class SequenceComparer : IEqualityComparer<Sequence>
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
            _specialTokens = [.. specialTokens.Keys];
            _compareLength = Math.Max(1, compareLength);
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

            int cx = 0, cy = 0;
            var xt = x.Tokens;
            var yt = y.Tokens;
            int xi = 0, yi = 0;
            while (xi < xt.Count && yi < yt.Count && cx < _compareLength && cy < _compareLength)
            {
                while (xi < xt.Count && _specialTokens.Contains(xt[xi])) xi++;
                while (yi < yt.Count && _specialTokens.Contains(yt[yi])) yi++;

                if (xi >= xt.Count || yi >= yt.Count)
                    break;

                if (xt[xi] != yt[yi])
                    return false;

                xi++; yi++;
                cx++; cy++;
            }

            return cx == cy;
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
                var hash = 17;
                var count = 0;
                var tokens = obj.Tokens;
                for (int i = 0; i < tokens.Count && count < _compareLength; i++)
                {
                    var t = tokens[i];
                    if (_specialTokens.Contains(t))
                        continue;

                    hash = hash * 23 + t.GetHashCode();
                    count++;
                }
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

    }
}