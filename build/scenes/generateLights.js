var htmlfile = WSH.CreateObject('htmlfile'), JSON;
htmlfile.write('<meta http-equiv="x-ua-compatible" content="IE=11" />');
htmlfile.close(JSON = htmlfile.parentWindow.JSON);

function copyArr(arr) {
    var res = [];

    for (var i = 0; i < arr.length; ++i)
        res[i] = arr[i];

    return res;
}

function to0ifmin(v, o1, o2) {
    return v < o1 && v < o2 ? 0 : v;
}

function buildColorVector(r, g, b) {
    r = to0ifmin(r, g, b);
    g = to0ifmin(g, r, b);
    b = to0ifmin(b, g, r);

    return [r, g, b];
}

function normalize(vect, mv) {
    var res = copyArr(vect);
    var max = Math.max.apply(Math, vect);

    for (var i = 0; i < res.length; ++i)
        res[i] = mv * res[i] / max;

    return res;
}

function vectToFixed(vect, dec) {
    var res = copyArr(vect);

    for (var i = 0; i < res.length; ++i)
        res[i] = +res[i].toFixed(dec);

    return res;
}

function addObject(x, z) {
    var r = Math.random();
    var g = Math.random();
    var b = Math.random();

    return {
        type: "sphere",
        center: [x, 77, z],
        radius: 2.5,
        color: vectToFixed([r, g, b], 4),
        emit: vectToFixed(normalize(buildColorVector(r, g, b), 40), 3)
    };
}

function fillScene() {
    var lights = [];

    for (var i = 5; i < 126; i += 10)
        lights.push(addObject(5, i));

    for (var i = 5; i < 96; i += 10)
        lights.push(addObject(i, 5));

    for (var i = 5; i < 126; i += 10)
        lights.push(addObject(95, i));

    return lights;
}

WScript.Echo(JSON.stringify(fillScene()));