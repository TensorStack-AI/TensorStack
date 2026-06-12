using System.Windows;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for PlaceholderTextBox.xaml
    /// </summary>
    public partial class PlaceholderTextBox : BaseControl
    {
        public PlaceholderTextBox()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty TextProperty = DependencyProperty.Register(nameof(Text), typeof(string), typeof(PlaceholderTextBox));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(string), typeof(PlaceholderTextBox));
        public static readonly DependencyProperty PlaceholderMarginProperty = DependencyProperty.Register(nameof(PlaceholderMargin), typeof(Thickness), typeof(PlaceholderTextBox), new PropertyMetadata(new Thickness(4,2,0,0)));
        public static readonly DependencyProperty PlaceholderOpacityProperty = DependencyProperty.Register(nameof(PlaceholderOpacity), typeof(double), typeof(PlaceholderTextBox), new PropertyMetadata(0.7d));
        public static readonly DependencyProperty PlaceholderFontStyleProperty = DependencyProperty.Register(nameof(PlaceholderFontStyle), typeof(FontStyle), typeof(PlaceholderTextBox), new PropertyMetadata(FontStyles.Italic));

        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        public string Placeholder
        {
            get { return (string)GetValue(PlaceholderProperty); }
            set { SetValue(PlaceholderProperty, value); }
        }

        public Thickness PlaceholderMargin
        {
            get { return (Thickness)GetValue(PlaceholderMarginProperty); }
            set { SetValue(PlaceholderMarginProperty, value); }
        }

        public double PlaceholderOpacity
        {
            get { return (double)GetValue(PlaceholderOpacityProperty); }
            set { SetValue(PlaceholderOpacityProperty, value); }
        }

        public FontStyle PlaceholderFontStyle
        {
            get { return (FontStyle)GetValue(PlaceholderFontStyleProperty); }
            set { SetValue(PlaceholderFontStyleProperty, value); }
        }
    }
}
