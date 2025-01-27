// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.Common
{
    public record ModelOptimization : IEquatable<ModelOptimization>
    {
        private readonly Optimization _optimizationLevel;
        private readonly SortedDictionary<string, long> _dimensionOverrides;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelOptimization"/> class.
        /// </summary>
        public ModelOptimization() : this(Optimization.All) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelOptimization"/> class.
        /// </summary>
        /// <param name="freeDimensionOverrides">The free dimension overrides.</param>
        public ModelOptimization(Optimization optimizationLevel)
        {
            _optimizationLevel = optimizationLevel;
            _dimensionOverrides = new SortedDictionary<string, long>();
        }

        /// <summary>
        /// Gets the optimization level.
        /// </summary>
        /// <value>The optimization level.</value>
        public Optimization OptimizationLevel => _optimizationLevel;


        /// <summary>
        /// Gets the dimension overrides.
        /// </summary>
        /// <value>The dimension overrides.</value>
        public SortedDictionary<string, long> DimensionOverrides => _dimensionOverrides;


        /// <summary>
        // Indicates whether the current ModelOptimization is equal to another
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns><c>true</c> if equal, <c>false</c> otherwise.</returns>
        public virtual bool Equals(ModelOptimization other)
        {
            if (other is null)
                return false;

            return other.DimensionOverrides.SequenceEqual(DimensionOverrides);
        }


        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table.</returns>
        public override int GetHashCode()
        {
            return HashCode.Combine(DimensionOverrides.GetHashCode());
        }
    }
}
