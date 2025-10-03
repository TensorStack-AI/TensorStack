// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using TensorStack.Common.Tensor;

namespace TensorStack.Common
{
    /// <summary>
    /// ModelParameters class to manage model input and output parameters
    /// Implements the <see cref="IDisposable" />
    /// </summary>
    /// <seealso cref="IDisposable" />
    public sealed class ModelParameters : IDisposable
    {
        private readonly RunOptions _runOptions;
        private readonly ModelMetadata _metadata;
        private readonly ParameterCollection _inputs;
        private readonly ParameterCollection _outputs;
        private readonly OrtAllocator _allocator;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelParameters"/> class.
        /// </summary>
        public ModelParameters(ModelMetadata metadata, CancellationToken cancellationToken = default)
        {
            _metadata = metadata;
            _allocator = metadata.Allocator;
            _runOptions = new RunOptions();
            _inputs = new ParameterCollection();
            _outputs = new ParameterCollection();
            cancellationToken.Register(Cancel, true);
        }

        /// <summary>
        /// Gets the run options.
        /// </summary>
        public RunOptions RunOptions => _runOptions;

        /// <summary>
        /// Gets the input names.
        /// </summary>
        public IReadOnlyCollection<string> InputNames => _inputs.Names;

        /// <summary>
        /// Gets the output names.
        /// </summary>
        public IReadOnlyCollection<string> OutputNames => _outputs.Names;

        /// <summary>
        /// Gets the input values.
        /// </summary>
        public IReadOnlyCollection<OrtValue> InputValues => _inputs.Values;

        /// <summary>
        /// Gets the output values.
        /// </summary>
        public IReadOnlyCollection<OrtValue> OutputValues => _outputs.Values;

        /// <summary>
        /// Gets the input name values.
        /// </summary>
        public IReadOnlyDictionary<string, OrtValue> InputNameValues => _inputs.NameValues;

        /// <summary>
        /// Gets the output name values.
        /// </summary>
        public IReadOnlyDictionary<string, OrtValue> OutputNameValues => _outputs.NameValues;

        /// <summary>
        /// Gets the expected input parameter count.
        /// </summary>
        public int InputCount => _metadata.Inputs.Count;

        /// <summary>
        /// Gets the expected output parameter count.
        /// </summary>
        public int OutputCount => _metadata.Outputs.Count;


        /// <summary>
        /// Adds an input OrtValue
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddInput(OrtValue value, bool dispose = true)
        {
            _inputs.Add(GetNextInputMetadata(), value, dispose);
        }


        /// <summary>
        /// Adds an input OrtValue at the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        public void AddInput(int index, OrtValue value, bool dispose = true)
        {
            var metadata = _metadata.Inputs[index];
            _inputs.Add(metadata, value, dispose);
        }


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput<T>(TensorSpan<T> value) where T : unmanaged, INumber<T>
        {
            var metadata = GetNextInputMetadata();
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput<T>(int index, TensorSpan<T> value) where T : unmanaged, INumber<T>
        {
            var metadata = _metadata.Inputs[index];
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(TensorSpan<string> value)
        {
            var metadata = GetNextInputMetadata();
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(value));
        }


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, TensorSpan<string> value)
        {
            var metadata = _metadata.Inputs[index];
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(value));
        }


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(TensorSpan<bool> value)
        {
            var metadata = GetNextInputMetadata();
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, TensorSpan<bool> value)
        {
            var metadata = _metadata.Inputs[index];
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(TensorSpan<byte> value)
        {
            var metadata = GetNextInputMetadata();
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, TensorSpan<byte> value)
        {
            var metadata = _metadata.Inputs[index];
            _inputs.Add(metadata, metadata.CreateTensorOrtValue(_allocator.Info, value));
        }


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput<T>(Tensor<T> value) where T : unmanaged, INumber<T> => AddInput(value.AsTensorSpan());



        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput<T>(int index, Tensor<T> value) where T : unmanaged, INumber<T> => AddInput(index, value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(Tensor<string> value) => AddInput(value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, Tensor<string> value) => AddInput(index, value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(Tensor<bool> value) => AddInput(value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, Tensor<bool> value) => AddInput(index, value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(Tensor<byte> value) => AddInput(value.AsTensorSpan());


        /// <summary>
        /// Adds a tensor input at the specified index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value">The value.</param>
        public void AddInput(int index, Tensor<byte> value) => AddInput(index, value.AsTensorSpan());


        /// <summary>
        /// Adds a scalar input.
        /// </summary>
        /// <param name="value">The value.</param>
        public void AddScalarInput<T>(T value) where T : unmanaged, INumber<T>
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, metaData.CreateScalarOrtValue(_allocator.Info, value));
        }


        public void AddScalarInput(string value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, metaData.CreateScalarOrtValue(_allocator.Info, value));
        }


        public void AddScalarInput(bool value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, metaData.CreateScalarOrtValue(_allocator.Info, value));
        }


        public void AddScalarInput(byte value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, metaData.CreateScalarOrtValue(_allocator.Info, value));
        }



        /// <summary>
        /// Adds an output parameter with unknown output size.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        public void AddOutput()
        {
            _outputs.AddName(GetNextOutputMetadata());
        }


        /// <summary>
        /// Adds an output with the specified OrtValue.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddOutput(OrtValue value)
        {
            _outputs.Add(GetNextOutputMetadata(), value);
        }


        /// <summary>
        /// Adds an output at the index with the specified OrtValue.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        public void AddOutput(int index, OrtValue value)
        {
            var metadata = _metadata.Outputs[index];
            _outputs.Add(metadata, value);
        }


        /// <summary>
        /// Adds an output with the specified buffer size.
        /// </summary>
        /// <param name="bufferDimension">The buffer dimensions.</param>
        public void AddOutput(ReadOnlySpan<int> bufferDimension)
        {
            var metadata = GetNextOutputMetadata();
            _outputs.Add(metadata, metadata.CreateOutputBuffer(_allocator, bufferDimension));
        }


        /// <summary>
        /// Adds an output at the specified indexwith the specified buffer size.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="bufferDimension">The buffer dimension.</param>
        public void AddOutput(int index, ReadOnlySpan<int> bufferDimension)
        {
            var metadata = _metadata.Outputs[index];
            _outputs.Add(metadata, metadata.CreateOutputBuffer(_allocator, bufferDimension));
        }


        /// <summary>
        /// Dispose resources.
        /// </summary>
        public void Dispose()
        {
            _inputs?.Dispose();
            _outputs?.Dispose();
            _runOptions?.Dispose();
        }


        private NamedMetadata GetNextInputMetadata()
        {
            if (_inputs.Names.Count >= _metadata.Inputs.Count)
                throw new ArgumentOutOfRangeException($"Too Many Inputs - No Metadata found for input index {_inputs.Names.Count - 1}");

            return _metadata.Inputs[_inputs.Names.Count];
        }


        private NamedMetadata GetNextOutputMetadata()
        {
            if (_outputs.Names.Count >= _metadata.Outputs.Count)
                throw new ArgumentOutOfRangeException($"Too Many Outputs - No Metadata found for output index {_outputs.Names.Count}");

            return _metadata.Outputs[_outputs.Names.Count];
        }


        /// <summary>
        /// Cancel the inference session
        /// </summary>
        private void Cancel()
        {
            try
            {
                if (_runOptions.IsClosed)
                    return;

                if (_runOptions.IsInvalid)
                    return;

                if (_runOptions.Terminate == true)
                    return;

                _runOptions.Terminate = true;
            }
            catch (Exception)
            {
                throw new OperationCanceledException();
            }
        }
    }
}
