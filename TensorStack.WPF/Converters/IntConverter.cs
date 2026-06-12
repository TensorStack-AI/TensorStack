using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
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


    [ValueConversion(typeof(int), typeof(bool))]
    public class GreaterThanToBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is IComparable v)
            {
                var convertedParameter = System.Convert.ChangeType(parameter, value.GetType(), culture);
                if (v.CompareTo(convertedParameter) > 0)
                {
                    return true;
                }
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    [ValueConversion(typeof(int), typeof(bool))]
    public class GreaterOrEqualThanToBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is IComparable v)
            {
                var convertedParameter = System.Convert.ChangeType(parameter, value.GetType(), culture);
                if (v.CompareTo(convertedParameter) >= 0)
                {
                    return true;
                }
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    [ValueConversion(typeof(int), typeof(bool))]
    public class LessThanToBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is IComparable v)
            {
                var convertedParameter = System.Convert.ChangeType(parameter, value.GetType(), culture);
                if (v.CompareTo(convertedParameter) < 0)
                {
                    return true;
                }
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }


    [ValueConversion(typeof(int), typeof(bool))]
    public class LessOrEqualThanToBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is IComparable v)
            {
                var convertedParameter = System.Convert.ChangeType(parameter, value.GetType(), culture);
                if (v.CompareTo(convertedParameter) <= 0)
                {
                    return true;
                }
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }



    public class FramesToTimeSpanConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            // Check for nulls and ensure we have both values
            if (values.Length < 2 || values[0] is not int frames || values[1] is not float frameRate)
                return TimeSpan.Zero;

            // Avoid division by zero
            if (frameRate <= 0) return TimeSpan.Zero;

            // Cast to double to avoid integer truncation
            double totalSeconds = (double)frames / frameRate;

            return TimeSpan.FromSeconds(totalSeconds);
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            return null; // Not needed
        }
    }


    [ValueConversion(typeof(object), typeof(int))]
    public class DecimalToPercentConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null) return 0;

            try
            {
                double decimalValue = System.Convert.ToDouble(value);
                double percentage = decimalValue * 100;
                return Math.Clamp((int)Math.Round(percentage), 0, 100);
            }
            catch (Exception)
            {
                return 0;
            }
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return 0.0;

            try
            {
                double intValue = System.Convert.ToDouble(value);
                double result = intValue / 100.0;
                if (targetType == typeof(float))
                {
                    return (float)result;
                }

                return result;
            }
            catch (Exception)
            {
                return 0.0;
            }
        }
    }


    [ValueConversion(typeof(int), typeof(bool))]
    public class AddIntConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int target)
            {
                if (parameter is int addition || int.TryParse(parameter.ToString(), out addition))
                    return target + addition;
            }
            else if (value is long lValue)
            {
                if (parameter is long addition || long.TryParse(parameter.ToString(), out addition))
                    return lValue + addition;
            }
            else if (value is float fValue)
            {
                if (parameter is float addition || float.TryParse(parameter.ToString(), out addition))
                    return fValue + addition;
            }
            else if (value is double dValue)
            {
                if (parameter is double addition || double.TryParse(parameter.ToString(), out addition))
                    return dValue + addition;
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    [ValueConversion(typeof(int), typeof(bool))]
    public class SubtractIntConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int iValue)
            {
                if (parameter is int subtraction || int.TryParse(parameter.ToString(), out subtraction))
                    return iValue - subtraction;
            }
            else if (value is long lValue)
            {
                if (parameter is long subtraction || long.TryParse(parameter.ToString(), out subtraction))
                    return lValue - subtraction;
            }
            else if (value is float fValue)
            {
                if (parameter is float subtraction || float.TryParse(parameter.ToString(), out subtraction))
                    return fValue - subtraction;
            }
            else if (value is double dValue)
            {
                if (parameter is double subtraction || double.TryParse(parameter.ToString(), out subtraction))
                    return dValue - subtraction;
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }


    public class IntArrayCommaConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int[] array)
            {
                return string.Join(',', array);
            }
            return string.Empty;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string text)
            {
                var intArray = text
                    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(x => int.TryParse(x, out _))
                    .Select(int.Parse)
                    .ToArray();
                return intArray;
            }
            return Array.Empty<int>();
        }
    }
}
