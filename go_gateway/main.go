package main

import (
	"bytes"
	"io"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	// Create the Gin server
	r := gin.Default()

	// Create the /predict endpoint for customers
	r.POST("/predict", func(c *gin.Context) {
		// 1. Read the JSON data sent by the customer
		body, _ := io.ReadAll(c.Request.Body)

		// 2. Set the address of our Python "Brain"
		// We use an environment variable so we can change it easily in production
		pythonURL := os.Getenv("ML_SERVICE_URL")
		if pythonURL == "" {
			pythonURL = "http://localhost:8000/predict"
		}

		// 3. Forward the request to the Python FastAPI service
		resp, err := http.Post(pythonURL, "application/json", bytes.NewBuffer(body))

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "ML Service is not reachable"})
			return
		}
		defer resp.Body.Close()

		// 4. Read the answer from Python and send it back to the customer
		responseBody, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, "application/json", responseBody)
	})

	// Start the Gateway on port 5000
	r.Run(":5000")
}
