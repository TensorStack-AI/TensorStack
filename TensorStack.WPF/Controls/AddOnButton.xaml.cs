using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for AddOnButton.xaml
    /// </summary>
    public partial class AddOnButton : BaseControl
    {
        public AddOnButton()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty CommandProperty = DependencyProperty.Register(nameof(Command), typeof(ICommand), typeof(AddOnButton));
        public static readonly DependencyProperty CommandParameterProperty = DependencyProperty.Register(nameof(CommandParameter), typeof(object), typeof(AddOnButton));
        public static readonly DependencyProperty IconProperty = DependencyProperty.Register(nameof(Icon), typeof(string), typeof(AddOnButton));
        public static readonly DependencyProperty IconSizeProperty = DependencyProperty.Register(nameof(IconSize), typeof(int), typeof(AddOnButton), new PropertyMetadata(16));
        public static readonly DependencyProperty IconColorProperty = DependencyProperty.Register(nameof(IconColor), typeof(Brush), typeof(AddOnButton), new PropertyMetadata(Brushes.Black));
        public static readonly DependencyProperty IconStyleProperty = DependencyProperty.Register(nameof(IconStyle), typeof(FontAwesomeIconStyle), typeof(AddOnButton), new PropertyMetadata(FontAwesomeIconStyle.SharpLight));
        public static readonly DependencyProperty IconMarginProperty = DependencyProperty.Register(nameof(IconMargin), typeof(Thickness), typeof(AddOnButton), new PropertyMetadata(new Thickness(0)));

        public ICommand Command
        {
            get { return (ICommand)GetValue(CommandProperty); }
            set { SetValue(CommandProperty, value); }
        }

        public object CommandParameter
        {
            get { return (object)GetValue(CommandParameterProperty); }
            set { SetValue(CommandParameterProperty, value); }
        }

        public int IconSize
        {
            get { return (int)GetValue(IconSizeProperty); }
            set { SetValue(IconSizeProperty, value); }
        }

        public string Icon
        {
            get { return (string)GetValue(IconProperty); }
            set { SetValue(IconProperty, value); }
        }

        public Brush IconColor
        {
            get { return (Brush)GetValue(IconColorProperty); }
            set { SetValue(IconColorProperty, value); }
        }

        public FontAwesomeIconStyle IconStyle
        {
            get { return (FontAwesomeIconStyle)GetValue(IconStyleProperty); }
            set { SetValue(IconStyleProperty, IconStyle); }
        }

        public Thickness IconMargin
        {
            get { return (Thickness)GetValue(IconMarginProperty); }
            set { SetValue(IconMarginProperty, value); }
        }


        private double _iconOpacity = 1f;
        public double IconOpacity
        {
            get { return _iconOpacity; }
            set { SetProperty(ref _iconOpacity, value); }
        }

    }
}
