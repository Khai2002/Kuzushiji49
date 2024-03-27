var canvas = new fabric.Canvas('drawingCanvas');

document.getElementById('submitButton').addEventListener('click', function() {
   var dataURL = canvas.toDataURL({ format: 'png' });
   // Send the dataURL to the server
});