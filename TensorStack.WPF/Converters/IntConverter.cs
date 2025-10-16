using System;
using System.Diagnostics;
using System.Globalization;
using System.Windows.Data;

namespace TensorStack.WPF.Converters
{
    [ValueConversion(typeof(string), typeof(Uri))]
    public class StringToIntConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            Debug.WriteLine(value);
            if (value is int)
                return value;

            string stringToConvert = (string)value;
            if (string.IsNullOrEmpty(stringToConvert))
            {
                return 0;
            }
            if (stringToConvert.Equals("-"))
            {
                return 0;
            }
            return int.Parse(stringToConvert);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            //Uri uriToConvertBack = (Uri)value;
            //if (uriToConvertBack != null && !uriToConvertBack.Equals(nullUri))
            //{
            //    return uriToConvertBack.OriginalString;
            //}

            return null;
        }
    }
}
