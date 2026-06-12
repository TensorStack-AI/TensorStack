using System;
using System.Globalization;
using System.IO;
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
                if (Path.Exists(stringToConvert) && !Path.IsPathFullyQualified(stringToConvert))
                    stringToConvert = Path.GetFullPath(stringToConvert);

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


    [ValueConversion(typeof(string), typeof(string))]
    public class FullPathToFileNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not string fullPath)
                return value;

            return Path.GetFileName(fullPath);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }


    [ValueConversion(typeof(string), typeof(string))]
    public class FullPathToFolderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not string fullPath)
                return value;

            return Path.GetDirectoryName(fullPath);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }


    public class StringArrayConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string[] array)
            {
                return string.Join(Environment.NewLine, array);
            }
            return string.Empty;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string text)
            {
                return text.Split(new[] { Environment.NewLine, "\n", "\r" }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            }
            return Array.Empty<string>();
        }
    }


    public class StringArrayCommaConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string[] array)
            {
                return string.Join(',', array);
            }
            return string.Empty;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string text)
            {
                return text.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            }
            return Array.Empty<string>();
        }
    }
}
