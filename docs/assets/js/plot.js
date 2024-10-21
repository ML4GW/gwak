let go_fetch_old=(filename, folder, title)=>{
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
            tickvals: [-1.617, -2.225, -2.985, -3.091],
            ticktext: ['1/2 days', '1/week', '1/month', '1/year']        }
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

let go_fetch=(filename, folder, title)=>{
fetch(filename).then((res) => res.json()).then((data) => {

const layout = {
        shapes: [
        {
            type: 'line',
            xref: 'paper',
            x0: 0.1,
            y0: -1.75,
            x1: 1,
            y1: -1.75,
            line:{
                color: '#E5659D',
                width: 2,
            },
        },
        {
            type: 'line',
            xref: 'paper',
            x0: 0.11,
            y0: -4.57,
            x1: 1,
            y1: -4.57,
            line:{
                color: '#E5659D',
                width: 2,
            },
        },
        {
            type: 'line',
            xref: 'paper',
            x0: 0.12,
            y0: -6.42,
            x1: 1,
            y1: -6.42,
            line:{
                color: '#E5659D',
                width: 2,
            },
        },
        ],
            annotations: [
                {
                    xref: 'paper',
                    x: 0,  // Place at the end of the line (adjust as needed)
                    y: -1.75,  // Same y-coordinate as the line
                    text: 'Cat2 1/day',  // The label for the shape
                    showarrow: false,
                    font: {
                        size: 14,
                        color: '#E5659D',  // Same color as the line
                    },
                    align: 'left'
                },
                {
                    xref: 'paper',
                    x: 0,  // Place at the end of the line (adjust as needed)
                    y: -4.57,  // Same y-coordinate as the line
                    text: 'Cat2 1/week',  // The label for the shape
                    showarrow: false,
                    font: {
                        size: 14,
                        color: '#E5659D',  // Same color as the line
                    },
                    align: 'left'
                },
                {
                    xref: 'paper',
                    x: 0,  // Place at the end of the line (adjust as needed)
                    y: -6.42,  // Same y-coordinate as the line
                    text: 'Cat2 1/month',  // The label for the shape
                    showarrow: false,
                    font: {
                        size: 14,
                        color: '#E5659D',  // Same color as the line
                    },
                    align: 'left'
                },
            ],
        title: title,
        width: 1100,
        height: 650,
        hovermode: 'closest',
        xaxis: {
            title: "Date"
        },
        yaxis: {
            title: "FAR",
            tickvals: [-1.065, -2.153, -3.299, -4.849, -6.725, -8.498],
            ticktext: ['1/month', '1/year', '1/10 years', '1/100 years',
                '1/1000 years', '1/10000 years']
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

let go_fetch_new=(filename, folder, title)=>{
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
            tickvals: [-1.325, -2.075, -2.675, -3.8, -5.025, -6.425, -7.545],
            ticktext: ['1/day', '1/week', '1/month', '1/year', '1/10 years', '1/100 years', '1/1000 years']        }
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
//     document.getElementsByClassName("button")[0].innerHTML = "O3a old"
//     go_fetch_old('all_O3a_spectrogram_old.json', 'all_O3a_spectrogram_old', 'O3a GWAK Detections before Heuristics')
// })
// document.getElementsByClassName("button")[1].addEventListener('click', function(){
//     document.getElementsByClassName("button")[1].innerHTML = "O3a new pearson"
//     go_fetch('all_O3a_spectrogram_pearson.json', 'all_O3a_spectrogram_pearson', 'O3a GWAK Detections using Pearson and Heuristics')
// })
// document.getElementsByClassName("button")[1].addEventListener('click', function(){
//     document.getElementsByClassName("button")[1].innerHTML = "O3a analysis"
//     go_fetch('all_O3a_spectrogram.json', 'all_O3a_spectrogram', 'O3a GWAK Detections no Pearson but with Heuristics')
// })
// document.getElementsByClassName("button")[0].addEventListener('click', function(){
//     document.getElementsByClassName("button")[0].innerHTML = "O3a analysis"
//     go_fetch_new('all_O3a_spectrogram_paper.json', 'all_O3a_spectrogram_paper', 'O3a GWAK Detections')
// })
document.getElementsByClassName("button")[0].addEventListener('click', function(){
    document.getElementsByClassName("button")[0].innerHTML = "O3a analysis"
    go_fetch('all_O3a_spectrogram_boom.json', 'all_O3a_spectrogram_boom', 'O3a GWAK Detections')
})
document.getElementsByClassName("button")[1].addEventListener('click', function(){
    document.getElementsByClassName("button")[1].innerHTML = "O3b analysis"
    go_fetch('all_O3b_spectrogram_new.json', 'all_O3b_spectrogram_new', 'O3b GWAK Detections')
})
document.getElementsByClassName("button")[2].addEventListener('click', function(){
    document.getElementsByClassName("button")[2].innerHTML = "Burst O3a training"
    go_fetch('burst_trainingO3a.json', 'burst_trainingO3a', 'Burst GWAK Detections O3a training')
})
document.getElementsByClassName("button")[3].addEventListener('click', function(){
    document.getElementsByClassName("button")[3].innerHTML = "Burst O3b training"
    go_fetch('burst_trainingO3b.json', 'burst_trainingO3b', 'Burst GWAK Detections O3b training')
})
document.getElementsByClassName("button")[0].innerHTML = "O3a analysis"
go_fetch('all_O3a_spectrogram_boom.json', 'all_O3a_spectrogram_boom', 'O3a GWAK Detections')


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
