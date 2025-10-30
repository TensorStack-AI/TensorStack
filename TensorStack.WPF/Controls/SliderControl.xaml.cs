using System;
using System.Windows;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for SliderControl.xaml
    /// </summary>
    public partial class SliderControl : BaseControl
    {
        private string _valueText;
        private string _valueFormat;
        private string _valuePostFix = string.Empty;
        private bool _isPercent;

        public SliderControl()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty TextProperty = DependencyProperty.Register(nameof(Text), typeof(string), typeof(SliderControl));
        public static readonly DependencyProperty ValueProperty = DependencyProperty.Register(nameof(Value), typeof(double), typeof(SliderControl), new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));
        public static readonly DependencyProperty MaximumProperty = DependencyProperty.Register(nameof(Maximum), typeof(double), typeof(SliderControl));
        public static readonly DependencyProperty MinimumProperty = DependencyProperty.Register(nameof(Minimum), typeof(double), typeof(SliderControl));
        public static readonly DependencyProperty TickFrequencyProperty = DependencyProperty.Register(nameof(TickFrequency), typeof(double), typeof(SliderControl));

        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        public double Value
        {
            get { return (double)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }

        public double Maximum
        {
            get { return (double)GetValue(MaximumProperty); }
            set { SetValue(MaximumProperty, value); }
        }

        public double Minimum
        {
            get { return (double)GetValue(MinimumProperty); }
            set { SetValue(MinimumProperty, value); }
        }

        public double TickFrequency
        {
            get { return (double)GetValue(TickFrequencyProperty); }
            set { SetValue(TickFrequencyProperty, value); }
        }

        public bool IsPercent
        {
            get { return _isPercent; }
            set { SetProperty(ref _isPercent, value); }
        }

        public string ValueText
        {
            get { return _valueText; }
            set { SetProperty(ref _valueText, value); }
        }

        public string ValueFormat
        {
            get { return _valueFormat; }
            set
            {
                SetProperty(ref _valueFormat, value);
                UpdateValueText();
            }
        }

        public string ValuePostFix
        {
            get { return _valuePostFix; }
            set
            {
                SetProperty(ref _valuePostFix, value);
                UpdateValueText();
            }
        }


        private void UpdateValueText()
        {
            if (_isPercent)
            {
                ValueText = $"{Math.Round(Value * 100.0):F0}%";
                return;
            }

            if (string.IsNullOrEmpty(_valueFormat))
            {
                ValueText = Value.ToString() + _valuePostFix;
                return;
            }

            ValueText = Value.ToString(_valueFormat) + _valuePostFix;
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            UpdateValueText();
        }
    }
}
