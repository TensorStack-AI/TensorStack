using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TensorStack.Image;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for TransparentImageElement.xaml
    /// </summary>
    public partial class TransparentImageElement : BaseControl
    {
        private int _decodePixelWidth;
        private int _decodePixelHeight;

        public TransparentImageElement()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty UriSourceProperty = DependencyProperty.Register(nameof(UriSource), typeof(Uri), typeof(TransparentImageElement), new PropertyMetadata<TransparentImageElement>((c) => c.OnUriSourceChanged()));
        public static readonly DependencyProperty BitmapSourceProperty = DependencyProperty.Register(nameof(BitmapSource), typeof(BitmapSource), typeof(TransparentImageElement), new PropertyMetadata<TransparentImageElement>((c) => c.OnBitmapSourceChanged()));
        public static readonly DependencyProperty StretchProperty = DependencyProperty.Register(nameof(Stretch), typeof(Stretch), typeof(TransparentImageElement), new PropertyMetadata(Stretch.Uniform));

        public Uri UriSource
        {
            get { return (Uri)GetValue(UriSourceProperty); }
            set { SetValue(UriSourceProperty, value); }
        }

        public BitmapSource BitmapSource
        {
            get { return (BitmapSource)GetValue(BitmapSourceProperty); }
            set { SetValue(BitmapSourceProperty, value); }
        }


        public Stretch Stretch
        {
            get { return (Stretch)GetValue(StretchProperty); }
            set { SetValue(StretchProperty, value); }
        }

        public int DecodePixelWidth
        {
            get { return _decodePixelWidth; }
            set { SetProperty(ref _decodePixelWidth, value); }
        }

        public int DecodePixelHeight
        {
            get { return _decodePixelHeight; }
            set { SetProperty(ref _decodePixelHeight, value); }
        }


        private async Task OnUriSourceChanged()
        {
            ImageControl.Source = !File.Exists(UriSource.AbsolutePath)
                ? default
                : await ImageService.LoadFromFileAsync(UriSource.AbsolutePath, _decodePixelWidth, _decodePixelHeight);
        }


        private Task OnBitmapSourceChanged()
        {
            ImageControl.Source = BitmapSource;
            return Task.CompletedTask;
        }
    }
}
