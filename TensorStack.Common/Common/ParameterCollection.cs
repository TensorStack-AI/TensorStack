// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace TensorStack.Common
{
    public sealed class ParameterCollection : IDisposable
    {
        private readonly List<NamedMetadata> _metaData;
        private readonly Dictionary<string, OrtValue> _values;
        private readonly List<string> _disposables;

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterCollection"/> class.
        /// </summary>
        public ParameterCollection()
        {
            _metaData = new List<NamedMetadata>();
            _values = new Dictionary<string, OrtValue>();
            _disposables = new List<string>();
        }


        /// <summary>
        /// Adds the specified NamedMetadata and OrtValue
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void Add(NamedMetadata metaData, OrtValue value, bool dispose = true)
        {
            _metaData.Add(metaData);
            _values.Add(metaData.Name, value);
            if (dispose)
            {
                _disposables.Add(metaData.Name);
            }
        }


        /// <summary>
        /// Adds the name only.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        public void AddName(NamedMetadata metaData, bool dispose = true)
        {
            _metaData.Add(metaData);
            _values.Add(metaData.Name, default);
            if (dispose)
            {
                _disposables.Add(metaData.Name);
            }
        }

        /// <summary>
        /// Gets the names.
        /// </summary>
        public IReadOnlyCollection<string> Names => _values.Keys;


        /// <summary>
        /// Gets the values.
        /// </summary>
        public IReadOnlyCollection<OrtValue> Values => _values.Values;


        /// <summary>
        /// Gets the name values.
        /// </summary>
        public IReadOnlyDictionary<string, OrtValue> NameValues => _values;


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var value in _values)
            {
                if (!_disposables.Contains(value.Key))
                    continue;

                value.Value?.Dispose();
            }

            _values.Clear();
            _metaData.Clear();
        }
    }
}
