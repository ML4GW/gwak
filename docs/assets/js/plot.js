let go_fetch_all=(filename, folder, title)=>{
fetch(filename).then((res) => res.json()).then((data) => {

const layout = {
        title: title,
        width: 1100,
        height: 650,
        hovermode: 'closest',
        xaxis: {
            title: "Date"
        },
        yaxis: {
            title: "FAR",
            tickvals: [-195],
            ticktext: ['1/year']                 }
    };

    Plotly.newPlot('plot', data, layout);

    document.getElementById('plot').on('plotly_click', function(data){

        var pts = '';
        for(var i=0; i < data.points.length; i++){
            pts = 'x = '+data.points[i].x +'\ny = '+
                data.points[i].y.toPrecision(4) + '\n\n';
        }

        show_image(`${folder}/${data.points[0].data.file[data.points[0].pointNumber]}`);

    });
});
}

let go_fetch_cat2=(filename, folder, title)=>{
fetch(filename).then((res) => res.json()).then((data) => {

const layout = {
        title: title,
        width: 1100,
        height: 650,
        hovermode: 'closest',
        xaxis: {
            title: "Date"
        },
        yaxis: {
            title: "FAR",
            tickvals: [-200],
            ticktext: ['1/year']
        }
    };

    Plotly.newPlot('plot', data, layout);


    document.getElementById('plot').on('plotly_click', function(data){

        var pts = '';
        for(var i=0; i < data.points.length; i++){
            pts = 'x = '+data.points[i].x +'\ny = '+
                data.points[i].y.toPrecision(4) + '\n\n';
        }

        show_image(`${folder}/${data.points[0].data.file[data.points[0].pointNumber]}`);

    });
});
}


document.getElementsByClassName("button")[0].addEventListener('click', function(){
    document.getElementsByClassName("button")[0].innerHTML = "HL MDC-short0"
    go_fetch_all('HL_short0-195.json', 'HL_short0-195', 'MDC O4b-2-short0 GWAK2.0 HL Detections')
})
document.getElementsByClassName("button")[1].addEventListener('click', function(){
    document.getElementsByClassName("button")[1].innerHTML = "HL MDC-short1"
    go_fetch_all('HL_short1-195.json', 'HL_short1-195', 'MDC O4b-2-short1 GWAK2.0 HL Detections')
})
document.getElementsByClassName("button")[2].addEventListener('click', function(){
    document.getElementsByClassName("button")[2].innerHTML = "HV MDC-short0"
    go_fetch_all('HV_short0-200.json', 'HV_short0-200', 'MDC O4b-2-short0 GWAK2.0 HV Detections')
})
document.getElementsByClassName("button")[3].addEventListener('click', function(){
    document.getElementsByClassName("button")[3].innerHTML = "HV MDC-short1"
    go_fetch_all('HV_short1-200.json', 'HV_short1-200', 'MDC O4b-2-short1 GWAK2.0 HV Detections')
})

document.getElementsByClassName("button")[0].innerHTML = "HL MDC-short0"
go_fetch_all('HL_short0-195.json', 'HL_short0-195', 'MDC O4b-2-short0 GWAK2.0 HL Detections')


function show_image(src){
    const backdrop = document.createElement('div');
    backdrop.classList.add('backdrop');
    backdrop.addEventListener('click', function(){
        this.remove();
    });

    const img = document.createElement('img');
    img.setAttribute('src', src);

    backdrop.appendChild(img);
    document.body.appendChild(backdrop);
}
