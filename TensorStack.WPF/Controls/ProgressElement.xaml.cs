using System;
using System.Windows;
using System.Windows.Controls;

namespace TensorStack.WPF.Controls
{
    public partial class ProgressElement : UserControl
    {
        /// <summary>Initializes a new instance of the <see cref="ProgressElement" /> class.</summary>
        public ProgressElement()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress), typeof(ProgressInfo), typeof(ProgressElement));

        public ProgressInfo Progress
        {
            get { return (ProgressInfo)GetValue(ProgressProperty); }
            set { SetValue(ProgressProperty, value); }
        }


        protected override Size MeasureOverride(Size constraint)
        {
            if (ProgressBarControl is not null)
                ProgressBarControl.Width = 0;

            return base.MeasureOverride(constraint);
        }


        protected override Size ArrangeOverride(Size arrangeBounds)
        {
            if (ProgressBarControl is not null)
                ProgressBarControl.Width = arrangeBounds.Width;

            return base.ArrangeOverride(arrangeBounds);
        }

    }


    public class ProgressInfo : BaseModel
    {
        private int _value;
        private int _maximum = 1;
        private string _message;

        public int Value
        {
            get { return _value; }
            set { SetProperty(ref _value, value); }
        }

        public int Maximum
        {
            get { return _maximum; }
            set { SetProperty(ref _maximum, value); }
        }

        public string Message
        {
            get { return _message; }
            set { SetProperty(ref _message, value); }
        }


        public void Update(string message = default)
        {
            Message = message;
        }


        public void Update(int value, int maximum, string message = default)
        {
            Value = value;
            Message = message;
            Maximum = Math.Max(1, maximum);
        }


        public void Indeterminate(string message = default)
        {
            Value = 0;
            Maximum = -1;
            Message = message;
        }


        public void Clear()
        {
            Value = 0;
            Maximum = 1;
            Message = null;
        }
    }
}
