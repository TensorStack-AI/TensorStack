using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for IconButton.xaml
    /// </summary>
    public partial class IconButton : BaseControl
    {
        public IconButton()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty TextProperty = DependencyProperty.Register(nameof(Text), typeof(string), typeof(IconButton));
        public static readonly DependencyProperty IsSelectedProperty = DependencyProperty.Register(nameof(IsSelected), typeof(bool), typeof(IconButton));
        public static readonly DependencyProperty CommandProperty = DependencyProperty.Register(nameof(Command), typeof(ICommand), typeof(IconButton));
        public static readonly DependencyProperty CommandParameterProperty = DependencyProperty.Register(nameof(CommandParameter), typeof(object), typeof(IconButton));
        public static readonly DependencyProperty OrientationProperty = DependencyProperty.Register(nameof(Orientation), typeof(Orientation), typeof(IconButton));
        public static readonly DependencyProperty PlacementProperty = DependencyProperty.Register(nameof(Placement), typeof(Dock), typeof(IconButton), new PropertyMetadata(Dock.Bottom));
        public static readonly DependencyProperty IconProperty = DependencyProperty.Register(nameof(Icon), typeof(string), typeof(IconButton));
        public static readonly DependencyProperty IconSizeProperty = DependencyProperty.Register(nameof(IconSize), typeof(int), typeof(IconButton), new PropertyMetadata(16));
        public static readonly DependencyProperty IconColorProperty = DependencyProperty.Register(nameof(IconColor), typeof(Brush), typeof(IconButton), new PropertyMetadata(Brushes.Black));
        public static readonly DependencyProperty IconStyleProperty = DependencyProperty.Register(nameof(IconStyle), typeof(FontAwesomeIconStyle), typeof(IconButton), new PropertyMetadata(FontAwesomeIconStyle.SharpLight));
        public static readonly DependencyProperty LineHeightProperty = DependencyProperty.Register(nameof(LineHeight), typeof(double), typeof(IconButton), new PropertyMetadata(double.NaN));
        public static readonly DependencyProperty IconMarginProperty = DependencyProperty.Register(nameof(IconMargin), typeof(Thickness), typeof(IconButton), new PropertyMetadata(new Thickness(0)));
        public static readonly DependencyProperty TextMarginProperty = DependencyProperty.Register(nameof(TextMargin), typeof(Thickness), typeof(IconButton), new PropertyMetadata(new Thickness(0)));
        public static readonly DependencyProperty AlignmentProperty = DependencyProperty.Register(nameof(Alignment), typeof(HorizontalAlignment), typeof(IconButton), new PropertyMetadata(HorizontalAlignment.Center));
        public static readonly DependencyProperty IconOpacityProperty = DependencyProperty.Register(nameof(IconOpacity), typeof(double), typeof(IconButton), new PropertyMetadata(1d));
        public static readonly DependencyProperty TextOpacityProperty = DependencyProperty.Register(nameof(TextOpacity), typeof(double), typeof(IconButton), new PropertyMetadata(1d));

        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        public bool IsSelected
        {
            get { return (bool)GetValue(IsSelectedProperty); }
            set { SetValue(IsSelectedProperty, value); }
        }

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

        public Orientation Orientation
        {
            get { return (Orientation)GetValue(OrientationProperty); }
            set { SetValue(OrientationProperty, value); }
        }
        public Dock Placement
        {
            get { return (Dock)GetValue(PlacementProperty); }
            set { SetValue(PlacementProperty, value); }
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

        public double LineHeight
        {
            get { return (double)GetValue(LineHeightProperty); }
            set { SetValue(LineHeightProperty, value); }
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

        public Thickness TextMargin
        {
            get { return (Thickness)GetValue(TextMarginProperty); }
            set { SetValue(TextMarginProperty, value); }
        }

        public HorizontalAlignment Alignment
        {
            get { return (HorizontalAlignment)GetValue(AlignmentProperty); }
            set { SetValue(AlignmentProperty, value); }
        }

        public double TextOpacity
        {
            get { return (double)GetValue(TextOpacityProperty); }
            set { SetValue(TextOpacityProperty, value); }
        }

        public double IconOpacity
        {
            get { return (double)GetValue(IconOpacityProperty); }
            set { SetValue(IconOpacityProperty, value); }
        }
          
    }
}
