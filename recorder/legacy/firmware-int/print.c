void print_integer(int value, FILE *fw)
{
    char result[INT_CONV_BUF_LEN];
    memset(result, 0, INT_CONV_BUF_LEN);

    if (value < 0) {
        putc('-', fw);
        value = -value;
    }

    for (int i = 0; i < INT_CONV_BUF_LEN && value > 0; value /= 10) {
        result[i++] = (value % 10) + '0';
    }

    for (int i = strlen(result) - 1; i >= 0; i--) {
        putc(result[i], fw);
    }
}

void print_float(float value, FILE *fw)
{
    char result[INT_CONV_BUF_LEN];

    if (value < 0) {
        putc('-', fw);
        value = -value;
    }

    value += 0.005;
    int integer = value;
    int fraction = (int)(value * 100) % 100;

    memset(result, 0, INT_CONV_BUF_LEN);

    int i = 0;
    result[i++] = (fraction % 10) + '0';
    result[i++] = (fraction / 10) + '0';
    result[i++] = '.';
    for (; i < INT_CONV_BUF_LEN && integer > 0; integer /= 10) {
        result[i++] = (integer % 10) + '0';
    }

    for (i = strlen(result) - 1; i >= 0; i--) {
        putc(result[i], fw);
    }
}
