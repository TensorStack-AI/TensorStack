// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Common.Tensor
{
    /// <summary>
    /// Tensor for long lifespan operations (heap allocated)
    /// Implements the <see cref="IDisposable" />
    /// </summary>
    /// <typeparam name="T">Tensor element type</typeparam>
    /// <seealso cref="IDisposable" />
    public class Tensor<T> : IDisposable
    {
        private Memory<T> _memory;
        private int[] _dimensions;
        private int[] _strides;

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor{T}"/> class.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="dimensions">The dimensions.</param>
        public Tensor(Memory<T> data, ReadOnlySpan<int> dimensions)
        {
            _memory = data;
            _dimensions = dimensions.ToArray();
            _strides = dimensions.GetStrides();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tensor{T}"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        public Tensor(ReadOnlySpan<int> dimensions)
            : this(new T[dimensions.GetProduct()], dimensions) { }


        /// <summary>
        /// Gets element type.
        /// </summary>
        public Type Type => typeof(T);

        /// <summary>
        /// Gets the tensor memory buffer.
        /// </summary>
        public Memory<T> Memory => _memory;

        /// <summary>
        /// Gets the tensor buffer span.
        /// </summary>
        public ReadOnlySpan<T> Span => _memory.Span;

        /// <summary>
        /// Gets the length.
        /// </summary>
        public long Length => _memory.Length;

        /// <summary>
        /// Gets the strides.
        /// </summary>
        public ReadOnlySpan<int> Strides => _strides;

        /// <summary>
        /// Gets the dimensions/shape.
        /// </summary>
        public ReadOnlySpan<int> Dimensions => _dimensions;


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
            get { return GetValue(indices.GetIndex(_strides)); }
            set { SetValue(indices.GetIndex(_strides), value); }
        }


        /// <summary>
        /// Gets the value with the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>T.</returns>
        public T GetValue(int index)
        {
            return Memory.Span[index];
        }


        /// <summary>
        /// Sets the value with the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        public void SetValue(int index, T value)
        {
            Memory.Span[index] = value;
        }


        /// <summary>
        /// Fills the tensor with the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        public void Fill(T value)
        {
            for (int i = 0; i < Length; i++)
            {
                SetValue(i, value);
            }
        }


        /// <summary>
        /// Clones this instance.
        /// </summary>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public Tensor<T> Clone()
        {
            return new Tensor<T>(Memory.ToArray(), Dimensions.ToArray());
        }


        /// <summary>
        /// Reshapes the Tensor with the specified dimensions.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        /// <exception cref="ArgumentException">Cannot reshape array due to mismatch in lengths - dimensions</exception>
        public void ReshapeTensor(ReadOnlySpan<int> dimensions)
        {
            var newSize = dimensions.GetProduct();
            if (newSize != Length)
                throw new ArgumentException($"Cannot reshape array due to mismatch in lengths", nameof(dimensions));

            _dimensions = dimensions.ToArray();
            _strides = dimensions.GetStrides();
        }


        /// <summary>
        /// Updates the the tensor data, Buffer.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        protected void UpdateTensor(Tensor<T> tensor)
        {
            _memory = tensor.Memory;
            _dimensions = tensor.Dimensions.ToArray();
            _strides = Dimensions.GetStrides();
            OnTensorDataChanged();
        }

        /// <summary>
        /// Called when Tensor data has changed
        /// </summary>
        protected virtual void OnTensorDataChanged()
        {
        }

        #region IDisposable

        private bool disposed = false;


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                _memory = null;
            }

            // Dispose unmanaged resources here (if any).
            disposed = true;
        }


        /// <summary>
        /// Finalizes an instance of the <see cref="ImageTensor"/> class.
        /// </summary>
        ~Tensor()
        {
            Dispose(false);
        }

        #endregion
    }
}
