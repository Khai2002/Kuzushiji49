document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let allPoints = [];  // Store all points for multiple shapes
    let points = [];     // Store points for the current shape

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    document.getElementById('clearBtn').addEventListener('click', clearCanvas);

    function startDrawing(e) {
        isDrawing = true;
        points = [];
        addPoint(e.clientX, e.clientY); // Adjust coordinates
    }

    function draw(e) {
        if (!isDrawing) return;

        addPoint(e.clientX, e.clientY); // Adjust coordinates
        drawSmoothLine();
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            allPoints.push([...points]);  // Store a copy of the points for the current shape
            points = [];  // Clear points for the current shape
        }
    }

    function addPoint(x, y) {
        const rect = canvas.getBoundingClientRect();
        points.push({ x: x - rect.left, y: y - rect.top });
    }

    function drawSmoothLine() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Set line attributes outside the loop
        ctx.lineWidth = 40;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000';

        allPoints.forEach(shapePoints => {
            if (shapePoints.length < 2) return;

            ctx.beginPath();
            ctx.moveTo(shapePoints[0].x, shapePoints[0].y);

            for (let i = 1; i < shapePoints.length - 2; i++) {
                const xc = (shapePoints[i].x + shapePoints[i + 1].x) / 2;
                const yc = (shapePoints[i].y + shapePoints[i + 1].y) / 2;
                ctx.quadraticCurveTo(shapePoints[i].x, shapePoints[i].y, xc, yc);
            }

            // For the last 2 points
            ctx.quadraticCurveTo(
                shapePoints[shapePoints.length - 2].x,
                shapePoints[shapePoints.length - 2].y,
                shapePoints[shapePoints.length - 1].x,
                shapePoints[shapePoints.length - 1].y
            );

            ctx.stroke();
        });

        // Draw the current shape in progress
        if (points.length > 1) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);

            for (let i = 1; i < points.length - 2; i++) {
                const xc = (points[i].x + points[i + 1].x) / 2;
                const yc = (points[i].y + points[i + 1].y) / 2;
                ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
            }

            // For the last 2 points
            ctx.quadraticCurveTo(
                points[points.length - 2].x,
                points[points.length - 2].y,
                points[points.length - 1].x,
                points[points.length - 1].y
            );

            ctx.stroke();
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        allPoints = [];
        points = [];
    }

    document.getElementById('submitBtn').addEventListener('click', function () {
    // Get the canvas element
    var canvas = document.getElementById('drawCanvas');

    // Get the image data as a data URL (PNG format)
    var imageDataUrl = canvas.toDataURL('image/png');

    // Create a new FormData object
    var formData = new FormData();

    // Append the image data to the FormData object
    formData.append('imageData', imageDataUrl);

    // Send a POST request to the server
    fetch('/process_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        // Handle the response from the server if needed
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

});
