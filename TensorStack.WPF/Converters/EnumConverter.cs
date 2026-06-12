// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace TensorStack.WPF.Converters
{
    [ValueConversion(typeof(Enum), typeof(string))]
    public class EnumDisplayNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Enum myEnum)
                return value;

            return myEnum.GetDisplayName();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return string.Empty;
        }
    }


    [ValueConversion(typeof(Enum), typeof(string))]
    public class EnumShortNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Enum myEnum)
                return value;

            return myEnum.GetShortName();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return string.Empty;
        }
    }


    [ValueConversion(typeof(Enum), typeof(string))]
    public class EnumDescriptionConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Enum myEnum)
                return value;

            return myEnum.GetDisplayDescription();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return string.Empty;
        }
    }


    [ValueConversion(typeof(Enum), typeof(string))]
    public class EnumDescriptionBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Enum myEnum)
                return false;

            if (string.IsNullOrEmpty(myEnum.GetDisplayDescription()))
                return false;

            return true;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return string.Empty;
        }
    }



    [ValueConversion(typeof(Enum), typeof(Visibility))]
    public class EnumToVisibilityConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return Visibility.Visible;

            return Visibility.Collapsed;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    [ValueConversion(typeof(Enum), typeof(Visibility))]
    public class EnumToHiddenConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return Visibility.Visible;

            return Visibility.Hidden;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    [ValueConversion(typeof(Enum), typeof(Visibility))]
    public class InverseEnumToVisibilityConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return Visibility.Collapsed;

            return Visibility.Visible;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    [ValueConversion(typeof(Enum), typeof(Visibility))]
    public class InverseEnumToHiddenConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return Visibility.Hidden;

            return Visibility.Visible;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    [ValueConversion(typeof(Enum), typeof(bool))]
    public class EnumToBooleanConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return true;

            return false;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    [ValueConversion(typeof(Enum), typeof(bool))]
    public class InverseEnumToBooleanConverter : EnumConverterBase
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (IsEnumValue(value, parameter))
                return false;

            return true;
        }

        public override object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }


    public abstract class EnumConverterBase : IValueConverter
    {
        public abstract object Convert(object value, Type targetType, object parameter, CultureInfo culture);
        public abstract object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture);

        protected static bool IsEnumValue(object value, object parameter)
        {
            if (parameter == null || value == null)
                return false;

            return parameter.ToString()
                .Split('|', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Contains(value.ToString(), StringComparer.OrdinalIgnoreCase);
        }
    }


    [ValueConversion(typeof(Enum), typeof(int))]
    public class EnumToIntConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Enum myEnum)
                return 0;

            return System.Convert.ToInt32(myEnum);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (!targetType.IsEnum)
                return null;

            if (Enum.IsDefined(targetType, value))
            {
                return Enum.ToObject(targetType, value);
            }
            return Enum.GetValues(targetType).GetValue(0);
        }
    }
}
