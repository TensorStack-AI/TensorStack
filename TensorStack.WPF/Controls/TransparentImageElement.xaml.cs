using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for TransparentImageElement.xaml
    /// </summary>
    public partial class TransparentImageElement : BaseControl
    {
        public TransparentImageElement()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(BitmapSource), typeof(TransparentImageElement));
        public static readonly DependencyProperty StretchProperty = DependencyProperty.Register(nameof(Stretch), typeof(Stretch), typeof(TransparentImageElement), new PropertyMetadata(Stretch.Uniform));

        public BitmapSource Source
        {
            get { return (BitmapSource)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
        }

        public Stretch Stretch
        {
            get { return (Stretch)GetValue(StretchProperty); }
            set { SetValue(StretchProperty, value); }
        }

    }
}
