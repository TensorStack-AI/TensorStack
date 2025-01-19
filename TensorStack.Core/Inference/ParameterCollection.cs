// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace TensorStack.Core.Inference
{
    public sealed class ParameterCollection : IDisposable
    {
        private readonly List<NamedMetadata> _metaData;
        private readonly Dictionary<string, OrtValue> _values;

        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterCollection"/> class.
        /// </summary>
        public ParameterCollection()
        {
            _metaData = new List<NamedMetadata>();
            _values = new Dictionary<string, OrtValue>();
        }


        /// <summary>
        /// Adds the specified NamedMetadata and OrtValue
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void Add(NamedMetadata metaData, OrtValue value)
        {
            _metaData.Add(metaData);
            _values.Add(metaData.Name, value);
        }


        /// <summary>
        /// Adds the name only.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        public void AddName(NamedMetadata metaData)
        {
            _metaData.Add(metaData);
            _values.Add(metaData.Name, default);
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
            foreach (var ortValue in _values.Values)
                ortValue?.Dispose();

            _values.Clear();
            _metaData.Clear();
        }
    }
}
