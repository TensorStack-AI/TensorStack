// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections;
using System.Collections.Generic;

namespace TensorStack.TextGeneration.Processing
{
    public class SequenceCollection : IEnumerable<Sequence>
    {
        private readonly List<Sequence> _sequences;

        /// <summary>
        /// Initializes a new instance of the <see cref="SequenceCollection"/> class.
        /// </summary>
        /// <param name="initalSize">Size of the inital.</param>
        public SequenceCollection(int initalSize = 1)
        {
            _sequences = new List<Sequence>(initalSize);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SequenceCollection"/> class.
        /// </summary>
        /// <param name="initialSequence">The initial sequence.</param>
        /// <param name="initalSize">Size of the inital.</param>
        public SequenceCollection(Sequence initialSequence, int initalSize = 1)
        {
            _sequences = new List<Sequence>(initalSize) { initialSequence };
        }

        /// <summary>
        /// Gets the count.
        /// </summary>
        public int Count => _sequences.Count;


        /// <summary>
        /// Adds the specified sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        public void Add(Sequence sequence)
        {
            if (_sequences.Contains(sequence))
                return;

            _sequences.Add(sequence);
        }


        /// <summary>
        /// Removes the specified sequences.
        /// </summary>
        /// <param name="sequences">The sequences.</param>
        public void Remove(SequenceCollection sequences)
        {
            foreach (var sequence in sequences)
            {
                _sequences.Remove(sequence);
            }
        }


        /// <summary>
        /// Removes the specified sequences.
        /// </summary>
        /// <param name="sequences">The sequences.</param>
        public void Remove(params Sequence[] sequences)
        {
            foreach (var sequence in sequences)
            {
                _sequences.Remove(sequence);
            }
        }


        /// <summary>
        /// Clears and dispose sequences.
        /// </summary>
        public void Clear()
        {
            _sequences.Dispose();
            _sequences.Clear();
        }

        /// <summary>
        /// Gets or sets the <see cref="Sequence"/> at the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>Sequence.</returns>
        public Sequence this[int index]
        {
            get { return _sequences[index]; }
            set { _sequences[index] = value; }
        }


        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<Sequence> GetEnumerator()
        {
            return _sequences.GetEnumerator();
        }


        /// <summary>
        /// Returns an enumerator that iterates through a collection.
        /// </summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator" /> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
