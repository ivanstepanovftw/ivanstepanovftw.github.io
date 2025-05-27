
Deps:
```shell
sudo apt install pdf2svg inkscape zopfli
```

Specs to comply:
- Математика рендерится на сервере в SVG + MathML.
- Математика копируется как TeX.
- SVG с `currentColor` добавленные при помощи `![alt](./img.svg)` должны менять цвет в тёмной теме.

Tests:

$$
a := b
$$

$$
a \colon= b
$$

$$
a \coloneqq b
$$

$$
a \coloneq b
$$

$$
a \equiv b
$$

$$
a \implies b
$$

$$
a \triangleq b
$$

$$
a \iff b
$$
