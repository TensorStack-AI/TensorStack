// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace TensorStack.Common
{
    internal class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T> where T : IDisposable
    {
        private bool _disposed;

        public DisposableList() { }

        public DisposableList(int count) 
            : base(count) { }

        public DisposableList(IEnumerable<T> collection) 
            : base(collection) { }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            
            if (disposing)
            {
                // Dispose in the reverse order (reverse order of creation)
                for (int i = Count - 1; i >= 0; --i)
                {
                    this[i]?.Dispose();
                }
                Clear();
                _disposed = true;
            }
        }


        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

    }
}
