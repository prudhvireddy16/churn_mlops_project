FROM golang:alpine
WORKDIR /app
COPY . .
RUN go mod download
RUN go build -o main .
EXPOSE 5000
CMD ["./main"]