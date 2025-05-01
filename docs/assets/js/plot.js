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
            tickvals: [-0.3899999999999997, -0.9100000000000001, -1.3899999999999997, -2.4299999999999997, -3.69, -5.13],
            ticktext: ['1/day', '1/week', '1/month', '1/year', '1/10 years', '1/100 years', '1/1000 years']                 }
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
            tickvals: [-3.3899999999999997, -6.25, -8.11, -9.75, -9.97, -9.99],
            ticktext: ['1/day', '1/week', '1/month', '1/year', '1/10 years', '1/100 years', '1/1000 years']
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


// document.getElementsByClassName("button")[0].addEventListener('click', function(){
//     document.getElementsByClassName("button")[0].innerHTML = "O3a analysis"
//     go_fetch('all_O3a_spectrogram_boom.json', 'all_O3a_spectrogram_boom', 'O3a GWAK Detections')
// })
// document.getElementsByClassName("button")[1].addEventListener('click', function(){
//     document.getElementsByClassName("button")[1].innerHTML = "O3b analysis"
//     go_fetch('all_O3b_spectrogram_new.json', 'all_O3b_spectrogram_new', 'O3b GWAK Detections')
// })
// document.getElementsByClassName("button")[2].addEventListener('click', function(){
//     document.getElementsByClassName("button")[2].innerHTML = "Burst O3a training"
//     go_fetch('burst_trainingO3a.json', 'burst_trainingO3a', 'Burst GWAK Detections O3a training')
// })
// document.getElementsByClassName("button")[3].addEventListener('click', function(){
//     document.getElementsByClassName("button")[3].innerHTML = "Burst O3b training"
//     go_fetch('burst_trainingO3b.json', 'burst_trainingO3b', 'Burst GWAK Detections O3b training')
// })

document.getElementsByClassName("button")[0].addEventListener('click', function(){
    document.getElementsByClassName("button")[0].innerHTML = "O3 GWAK Detections"
    go_fetch_all('all_O3.json', 'all_O3', 'O3 GWAK Detections')
})
document.getElementsByClassName("button")[1].addEventListener('click', function(){
    document.getElementsByClassName("button")[1].innerHTML = "O3 GWAK Detections NEW"
    go_fetch_all('all_O3_new.json', 'all_O3_new', 'O3 GWAK Detections NEW')
})
document.getElementsByClassName("button")[2].addEventListener('click', function(){
    document.getElementsByClassName("button")[2].innerHTML = "Category 2 GWAK Detections"
    go_fetch_cat2('all_O3_cat2.json', 'all_O3_cat2', 'Category 2 GWAK Detections')
})

document.getElementsByClassName("button")[0].innerHTML = "O3 GWAK Detections"
go_fetch_all('all_O3.json', 'all_O3', 'O3 GWAK Detections')


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
