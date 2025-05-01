const fs = require('fs');
const path = require('path');

const dataFolder = path.join(__dirname, 'HL_eventdisplay/short1-195');

const files = fs.readdirSync(dataFolder);

let x = [];
let y = [];
let f = [];

files.forEach((file) => {
    if(!file.match(/\d+_.+/)) return;
    const [name_, date, value, value2] = file.split('_');
    x.push(gpsTimeToDate(date));
    y.push(value2.replaceAll(".png", ""));
    f.push(file);
});

fs.writeFile(path.join(__dirname, 'HL_short1-195.json'), JSON.stringify([{x, y, file: f, mode: "markers"}], null, 2), (err) => {
    if (err) {
        console.log(err);
    }
});

function gpsTimeToDate(gpsTime) {
    const timeDifference = (new Date(1980, 0, 6) - new Date(1970, 0, 1)) + (gpsTime * 1000);
    return new Date(timeDifference);
}
