import React, { useRef, useState } from "react";

const DrawingCanvas = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [Prediction, setPrediction] = useState();
  // Start drawing
  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    const context = canvasRef.current.getContext("2d");
    context.strokeStyle = "white";
    context.lineWidth = 60;
    context.lineCap = "round"; // Set pen to round
    context.beginPath();
    context.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };
  // Draw on canvas
  const draw = ({ nativeEvent }) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = nativeEvent;
    const context = canvasRef.current.getContext("2d");
    context.lineTo(offsetX, offsetY);
    context.lineCap = "round"; // Set pen to round
    context.stroke();
  };

  // Stop drawing
  const stopDrawing = () => {
    setIsDrawing(false);
  };

  // Get pixel data
  const sendData = async () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    try {
      const response = await fetch("http://localhost:4000/send-data", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ data: Array.from(imageData.data) }),
      });
      const result = await response.json();
      console.log("Response from backend:", result);
      setPrediction(result.result);
      console.log(result.probability)
    } catch (error) {
      console.error("Error sending data:", error);
    }
  };
  
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        width="512 px"
        height="512 px"
        style={{ border: "1px solid white" }}
      />
      <br />
      <button onClick={sendData}>Get Result</button>
      <button onClick={clearCanvas}>Clear</button>
      <br />
      <h1>The number is : {Prediction}</h1>
    </div>
  );
};

export default DrawingCanvas;
