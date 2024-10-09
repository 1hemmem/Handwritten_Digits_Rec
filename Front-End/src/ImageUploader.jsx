import React, { useState, useRef } from "react";

const ImageUploader = () => {
  const [prediction, setPrediction] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      // Create a preview of the image
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);

      // Prepare the image data for sending to the backend
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:4000/send-data", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Response from backend:", result);
        setPrediction(result.result);
      } catch (error) {
        console.error("Error sending data:", error);
        setPrediction("Error occurred during prediction");
      }
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleImageUpload}
        accept="image/*"
        style={{ display: "none" }}
      />
      <button onClick={triggerFileInput}>Upload Image</button>
      {imagePreview && (
        <div>
          <img src={imagePreview} alt="Uploaded preview" style={{ maxWidth: "200px", maxHeight: "200px" }} />
        </div>
      )}
      {prediction && <h2>The prediction is: {prediction}</h2>}
    </div>
  );
};

export default ImageUploader;
