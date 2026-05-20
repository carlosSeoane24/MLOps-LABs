#!/bin/bash
echo "📦 Contenedores activos:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "📡 Ping al Backend (FastAPI)..."
curl -s http://localhost:8000/health
echo ""