// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Common.Tensor
{
    /// <summary>
    /// TensorSpan for short lifespan operations (stack allocated)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct TensorSpan<T>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TensorSpan{T}"/> struct.
        /// </summary>
        /// <param name="dataSpan">The data span.</param>
        /// <param name="dimensions">The dimensions.</param>
        public TensorSpan(Span<T> dataSpan, ReadOnlySpan<int> dimensions)
        {
            Span = dataSpan;
            Dimensions = dimensions;
            Strides = dimensions.GetStrides();
        }

        public TensorSpan(ReadOnlySpan<int> dimensions)
            : this(new T[dimensions.GetProduct()], dimensions) { }


        /// <summary>
        /// Gets element type.
        /// </summary>
        public Type Type => typeof(T);

        /// <summary>
        /// Gets the data span.
        /// </summary>
        public Span<T> Span { get; }

        /// <summary>
        /// Gets the length.
        /// </summary>
        public long Length => Span.Length;

        /// <summary>
        /// Gets the strides.
        /// </summary>
        public ReadOnlySpan<int> Strides { get; }

        /// <summary>
        /// Gets the dimensions.
        /// </summary>
        public ReadOnlySpan<int> Dimensions { get; }


        /// <summary>
        /// Gets or sets the <see cref="T"/> with the specified indices.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>T.</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public T this[params int[] indices]
        {
            get
            {
                ArgumentNullException.ThrowIfNull(indices);
                var span = new ReadOnlySpan<int>(indices);
                return this[span];
            }
            set
            {
                ArgumentNullException.ThrowIfNull(indices);
                var span = new ReadOnlySpan<int>(indices);
                this[span] = value;
            }
        }


        /// <summary>
        /// Gets or sets the <see cref="T"/> with the specified indices.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>T.</returns>
        public T this[ReadOnlySpan<int> indices]
        {
            get { return GetValue(indices.GetIndex(Strides)); }
            set { SetValue(indices.GetIndex(Strides), value); }
        }


        /// <summary>
        /// Gets the value with the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>T.</returns>
        public T GetValue(int index)
        {
            return Span[index];
        }


        /// <summary>
        /// Sets the value with the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        public void SetValue(int index, T value)
        {
            Span[index] = value;
        }

    }
}
