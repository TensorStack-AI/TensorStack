using System.Windows;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for ColorPicker.xaml
    /// </summary>
    public partial class ToggleButton : BaseControl
    {
        public ToggleButton()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty IsCheckedProperty =
            DependencyProperty.Register(nameof(IsChecked), typeof(bool), typeof(ToggleButton), new PropertyMetadata(false));

        public static readonly DependencyProperty IconTrueProperty =
            DependencyProperty.Register(nameof(IconTrue), typeof(string), typeof(ToggleButton), new PropertyMetadata("\uf00c"));

        public static readonly DependencyProperty IconFalseProperty =
            DependencyProperty.Register(nameof(IconFalse), typeof(string), typeof(ToggleButton), new PropertyMetadata("\uf00d"));

        public static readonly DependencyProperty SizeTrueProperty =
             DependencyProperty.Register(nameof(SizeTrue), typeof(int), typeof(ToggleButton), new PropertyMetadata(16));

        public static readonly DependencyProperty SizeFalseProperty =
            DependencyProperty.Register(nameof(SizeFalse), typeof(int), typeof(ToggleButton), new PropertyMetadata(16));

        public static readonly DependencyProperty ColorTrueProperty =
            DependencyProperty.Register(nameof(ColorTrue), typeof(System.Windows.Media.Brush), typeof(ToggleButton), new PropertyMetadata(System.Windows.Media.Brushes.Black));

        public static readonly DependencyProperty ColorFalseProperty =
            DependencyProperty.Register(nameof(ColorFalse), typeof(System.Windows.Media.Brush), typeof(ToggleButton), new PropertyMetadata(System.Windows.Media.Brushes.Black));

        public static readonly DependencyProperty IconStyleTrueProperty =
            DependencyProperty.Register(nameof(IconStyleTrue), typeof(FontAwesomeIconStyle), typeof(ToggleButton), new PropertyMetadata(FontAwesomeIconStyle.SharpLight));

        public static readonly DependencyProperty IconStyleFalseProperty =
            DependencyProperty.Register(nameof(IconStyleFalse), typeof(FontAwesomeIconStyle), typeof(ToggleButton), new PropertyMetadata(FontAwesomeIconStyle.SharpLight));

        public static readonly DependencyProperty IsSpinnerTrueProperty =
            DependencyProperty.Register(nameof(IsSpinnerTrue), typeof(bool), typeof(ToggleButton));
       
       public static readonly DependencyProperty IsSpinnerFalseProperty =
            DependencyProperty.Register(nameof(IsSpinnerFalse), typeof(bool), typeof(ToggleButton));


        public bool IsChecked
        {
            get { return (bool)GetValue(IsCheckedProperty); }
            set { SetValue(IsCheckedProperty, value); }
        }

        /// <summary>
        /// Gets or sets the icon.
        /// </summary>
        public string IconTrue
        {
            get { return (string)GetValue(IconTrueProperty); }
            set { SetValue(IconTrueProperty, value); }
        }

        public string IconFalse
        {
            get { return (string)GetValue(IconFalseProperty); }
            set { SetValue(IconFalseProperty, value); }
        }



        public int SizeTrue
        {
            get { return (int)GetValue(SizeTrueProperty); }
            set { SetValue(SizeTrueProperty, value); }
        }

        public int SizeFalse
        {
            get { return (int)GetValue(SizeFalseProperty); }
            set { SetValue(SizeFalseProperty, value); }
        }

        public System.Windows.Media.Brush ColorTrue
        {
            get { return (System.Windows.Media.Brush)GetValue(ColorTrueProperty); }
            set { SetValue(ColorTrueProperty, value); }
        }

        public System.Windows.Media.Brush ColorFalse
        {
            get { return (System.Windows.Media.Brush)GetValue(ColorFalseProperty); }
            set { SetValue(ColorFalseProperty, value); }
        }

        public FontAwesomeIconStyle IconStyleTrue
        {
            get { return (FontAwesomeIconStyle)GetValue(IconStyleTrueProperty); }
            set { SetValue(IconStyleTrueProperty, value); }
        }

        public FontAwesomeIconStyle IconStyleFalse
        {
            get { return (FontAwesomeIconStyle)GetValue(IconStyleFalseProperty); }
            set { SetValue(IconStyleFalseProperty, value); }
        }

        public bool IsSpinnerTrue
        {
            get { return (bool)GetValue(IsSpinnerTrueProperty); }
            set { SetValue(IsSpinnerTrueProperty, value); }
        }

        public bool IsSpinnerFalse
        {
            get { return (bool)GetValue(IsSpinnerFalseProperty); }
            set { SetValue(IsSpinnerFalseProperty, value); }
        }
    }
}
