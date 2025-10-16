using System;
using System.Globalization;
using System.Windows.Controls;
using System.Windows.Data;

namespace TensorStack.WPF.Converters
{
    [ValueConversion(typeof(string), typeof(Uri))]
    public class StringToUriConverter : IValueConverter
    {
        private readonly Uri nullUri = new Uri("about:blank");

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            string stringToConvert = (string)value;
            if (stringToConvert != null)
            {
                return new Uri(stringToConvert);
            }
            return nullUri;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            Uri uriToConvertBack = (Uri)value;
            if (uriToConvertBack != null && !uriToConvertBack.Equals(nullUri))
            {
                return uriToConvertBack.OriginalString;
            }

            return null;
        }
    }



    [ValueConversion(typeof(string), typeof(string))]
    public class StringCaseConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string stringValue && Enum.TryParse<CharacterCasing>(parameter.ToString(), out var casing))
            {
                switch (casing)
                {
                    case CharacterCasing.Lower:
                        return stringValue.ToLower();
                    case CharacterCasing.Normal:
                        return stringValue;
                    case CharacterCasing.Upper:
                        return stringValue.ToUpper();
                    default:
                        return stringValue;
                }
            }
            return string.Empty;
        }


        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
