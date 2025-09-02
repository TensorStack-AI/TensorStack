// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace TensorStack.Common
{
    public class MapCollection<TKey, TValue> : IDictionary<TKey, TValue>
    {
        private readonly IDictionary<TKey, TValue> _collectionKeys;
        private readonly IDictionary<TValue, TKey> _collectionValues;

        public MapCollection()
        {
            _collectionKeys = new Dictionary<TKey, TValue>();
            _collectionValues = new Dictionary<TValue, TKey>();
        }

        public MapCollection(IDictionary<TKey, TValue> collection) : this()
        {
            foreach (var item in collection)
                Add(item.Key, item.Value);
        }

        public MapCollection(IDictionary<TValue, TKey> collection) : this()
        {
            foreach (var item in collection)
                Add(item.Value, item.Key);
        }


        public ICollection<TKey> Keys => _collectionKeys.Keys;
        public ICollection<TValue> Values => _collectionValues.Keys;
        public int Count => _collectionKeys.Count;
        public bool IsReadOnly => false;


        public TValue this[TKey key]
        {
            get { return _collectionKeys[key]; }
            set
            {
                if (_collectionKeys.TryGetValue(key, out var oldValue))
                    _collectionValues.Remove(oldValue);

                _collectionKeys[key] = value;
                _collectionValues[value] = key;
            }
        }


        public TKey this[TValue key]
        {
            get { return _collectionValues[key]; }
            set
            {
                if (_collectionValues.TryGetValue(key, out var oldKey))
                    _collectionKeys.Remove(oldKey);

                _collectionValues[key] = value;
                _collectionKeys[value] = key;
            }
        }


        public void Add(TKey key, TValue value)
        {
            _collectionKeys.Add(key, value);
            _collectionValues.Add(value, key);
        }

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            _collectionKeys.Add(item.Key, item.Value);
            _collectionValues.Add(item.Value, item.Key);
        }

        public bool TryAdd(TKey key, TValue value)
        {
            return _collectionKeys.TryAdd(key, value)
                && _collectionValues.TryAdd(value, key);
        }

        public void Clear()
        {
            _collectionKeys.Clear();
            _collectionValues.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return _collectionKeys.Contains(item);
        }

        public bool ContainsKey(TKey key)
        {
            return _collectionKeys.ContainsKey(key);
        }

        public bool ContainsKey(TValue key)
        {
            return _collectionValues.ContainsKey(key);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            return _collectionKeys.GetEnumerator();
        }

        public bool Remove(TKey key)
        {
            if (_collectionKeys.TryGetValue(key, out TValue value))
            {
                _collectionKeys.Remove(key);
                _collectionValues.Remove(value);
                return true;
            }
            return false;
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return Remove(item.Key);
        }

        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
        {
            return _collectionKeys.TryGetValue(key, out value);
        }


        public bool TryGetValue(TValue key, [MaybeNullWhen(false)] out TKey value)
        {
            return _collectionValues.TryGetValue(key, out value);
        }


        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}