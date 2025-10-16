// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading.Tasks;
using System.Windows;

namespace TensorStack.WPF
{
    public class PropertyMetadata<T> : FrameworkPropertyMetadata
          where T : DependencyObject
    {
        public PropertyMetadata(Func<T, Task> function)
            : base(CreateCallback(function)) { }

        private static PropertyChangedCallback CreateCallback(Func<T, Task> function)
        {
            return async (d, e) =>
            {
                if (d is T control)
                    await function(control);
            };
        }
    }

    public class PropertyMetadata<T, U> : FrameworkPropertyMetadata
        where T : DependencyObject
        where U : class
    {
        public PropertyMetadata(Func<T, U, Task> function)
             : base(CreateCallback(function)) { }

        public PropertyMetadata(Func<T, U, U, Task> function)
           : base(CreateCallback(function)) { }

        private static PropertyChangedCallback CreateCallback(Func<T, U, Task> function)
        {
            return async (d, e) =>
            {
                if (d is T control)
                    await function(control, e.NewValue as U);
            };
        }

        private static PropertyChangedCallback CreateCallback(Func<T, U, U, Task> function)
        {
            return async (d, e) =>
            {
                if (d is T control)
                    await function(control, e.OldValue as U, e.NewValue as U);
            };
        }
    }
}
