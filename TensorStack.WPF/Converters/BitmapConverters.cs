using System;
using System.Globalization;
using System.Windows.Data;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;
using TensorStack.Image;

namespace TensorStack.WPF.Converters
{

    /// <summary>
    /// Converter for converting a boolean to its inverse value
    /// </summary>
    /// <seealso cref="System.Windows.Data.IValueConverter" />
    [ValueConversion(typeof(bool), typeof(bool))]
    public class ImageTensorToBitmapConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ImageTensor imageTensor)
            {
                return imageTensor.ToImage();
            }
            return null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }

    /// <summary>
    /// Converter for converting a boolean to its inverse value
    /// </summary>
    /// <seealso cref="System.Windows.Data.IValueConverter" />
    [ValueConversion(typeof(bool), typeof(bool))]
    public class VideoFrameToBitmapConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is VideoFrame frame)
            {
                return frame.Frame.ToImage();
            }
            return null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }
}
