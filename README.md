# PRISM - Microservices Failure Prediction Engine

A production-grade real-time failure prediction system for microservices monitoring, powered by machine learning and distributed tracing.

## 🎯 Features

### Core Capabilities
- **Real-time Monitoring**: Live telemetry ingestion from microservices
- **ML-Powered Predictions**: Predict service failures before they occur
- **Dependency Mapping**: Visualize service dependencies and failure propagation
- **Alert Management**: Real-time alerting with configurable thresholds
- **Circuit Breaker Pattern**: Graceful degradation on external service failures
- **Correlation Tracking**: Request tracing across distributed systems

### Production Features
- **High Availability**: Kubernetes-ready with health checks
- **Observability**: Prometheus metrics and Grafana dashboards
- **Security**: Helmet security headers, CORS, rate limiting, input validation
- **Resilience**: Retry logic with exponential backoff, circuit breakers
- **Error Handling**: Global error boundaries and structured error responses
- **Logging**: Structured logging with Winston and daily rotation
- **Testing**: Unit tests, integration tests, E2E ready
- **CI/CD**: GitHub Actions automation for testing and deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PRISM Frontend                         │
│         (React + Vite + Zustand + Recharts)             │
│                                                          │
│  • Error Boundaries  • Retry Logic  • Real-time UI      │
└─────────────────┬───────────────────────────────────────┘
                  │ WebSocket / REST API
┌─────────────────▼───────────────────────────────────────┐
│                  PRISM Backend (Express.js)             │
│  • Request Correlation  • Circuit Breaker               │
│  • Rate Limiting  • Metrics Collection                  │
│  • Graceful Shutdown  • WebSocket Server                │
└─────────────────┬───────────────────────────────────────┘
     ┌────────────┼────────────┐
     │            │            │
┌────▼──┐  ┌─────▼──┐  ┌─────▼──┐
│ PostgreSQL │ Redis  │ ML Service
│ Database  │ Cache  │ (Python)
└───────┘  └────────┘  └────────┘

Monitoring Stack:
Prometheus ──► Grafana (Dashboards)
     ▲
     │
Backend Metrics (/metrics endpoint)
```

## 📦 Tech Stack

### Backend
- **Runtime**: Node.js 20+
- **Framework**: Express.js 4.x
- **Database**: PostgreSQL 16 + Prisma ORM
- **Cache**: Redis 7 (optional)
- **Logging**: Winston 3.x
- **Monitoring**: Prometheus client
- **Message Queue**: Socket.io (real-time)

### Frontend
- **Runtime**: Node.js 20+
- **Framework**: React 19 + TypeScript
- **Build**: Vite 8.x
- **State**: Zustand 5.x
- **Charts**: Recharts 3.x
- **Routing**: React Router 7.x
- **Styling**: Tailwind CSS 4.x + PostCSS

### DevOps
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes-ready
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

## 🚀 Quick Start

### Prerequisites
- Node.js 20+
- Docker & Docker Compose
- PostgreSQL 16
- Redis 7 (optional)

### Local Development

1. **Clone and Setup**
```bash
git clone <repo>
cd PrismSystem
cp .env.example .env
```

2. **Install Dependencies**
```bash
# Backend
cd backend
npm install
npx prisma generate
npx prisma migrate dev

# Frontend
cd ../frontend
npm install
```

3. **Start Development Servers**
```bash
# Terminal 1: Backend
cd backend
npm run dev

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

### Docker Deployment

```bash
docker-compose up -d
```

Services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3002 (admin/admin)
- PostgreSQL: localhost:5432

## 📋 Environment Variables

### Backend (.env)

```env
# Server
NODE_ENV=production
PORT=3001

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/prism_db

# ML Service
ML_SERVICE_URL=http://localhost:8000
ML_SERVICE_TIMEOUT=10000
ML_SERVICE_RETRIES=3

# Logging
LOG_LEVEL=info

# CORS
CORS_ORIGIN=https://yourdomain.com

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000  # 15 minutes
RATE_LIMIT_MAX_REQUESTS=100

# WebSocket
WS_PING_INTERVAL=25000
WS_PING_TIMEOUT=20000
```

### Frontend (.env.local)

```env
VITE_API_BASE_URL=https://api.yourdomain.com/api/v1
```

## 🔌 API Endpoints

### Dashboard
- `GET /api/v1/dashboard/summary` - System overview
- `GET /api/v1/services` - List all services
- `GET /api/v1/service/:name` - Service details
- `GET /api/v1/services/dependencies` - Dependency graph

### Telemetry
- `POST /api/v1/telemetry` - Ingest telemetry
- `GET /api/v1/telemetry` - Query telemetry (paginated)

### Predictions
- `GET /api/v1/predictions` - List predictions
- `POST /api/v1/predictions` - Generate prediction

### Health & Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## 🧪 Testing

### Backend Unit Tests
```bash
cd backend
npm test
npm test -- --coverage
```

### Backend E2E Tests
```bash
npm run test:e2e
```

### Frontend Tests (Coming Soon)
```bash
cd frontend
npm test
```

## 📊 Monitoring

### Prometheus Metrics

**HTTP Metrics:**
- `http_requests_total` - Total requests by method, route, status
- `http_request_duration_seconds` - Request latency histogram
- `http_errors_total` - Total error requests

**Database Metrics:**
- `database_query_duration_seconds` - Query latency
- `database_errors_total` - Database errors

**Business Metrics:**
- `telemetry_points_ingested_total` - Telemetry ingestion rate
- `predictions_generated_total` - Prediction generation rate
- `prediction_confidence` - Prediction confidence distribution
- `circuit_breaker_state` - Circuit breaker status per service

### Grafana Dashboards
- System Health Dashboard
- Service Metrics Dashboard
- ML Prediction Dashboard
- Error Rate & Latency Dashboard

## 🔐 Security Features

- **Helmet**: HTTP security headers
- **CORS**: Configurable origin validation
- **Rate Limiting**: Token bucket algorithm
- **Input Validation**: express-validator + Prisma
- **Error Handling**: Secure error responses (no stack traces in production)
- **Logging**: Correlation IDs for audit trails
- **Dependencies**: Regular security updates via Dependabot

### Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **HTTPS**: Always use HTTPS in production
3. **Authentication**: Implement JWT/OAuth for API (coming soon)
4. **Database**: Use connection pooling and parameterized queries
5. **Secrets**: Use managed secrets in production (AWS Secrets Manager, etc.)

## 🏥 Health Checks

```bash
# Backend health
curl http://localhost:3001/health

# Frontend (from browser)
http://localhost:3000

# Metrics
curl http://localhost:3001/metrics
```

## 📈 Scaling & Deployment

### Kubernetes Deployment

```bash
# Build and push images
docker build -t your-registry/prism-backend backend/
docker build -t your-registry/prism-frontend frontend/
docker push your-registry/prism-backend
docker push your-registry/prism-frontend

# Deploy with Helm or kubectl
kubectl apply -f k8s/
```

### Performance Optimization

- Database query optimization and indexing
- Redis caching for frequently accessed data
- Frontend code splitting and lazy loading
- Gzip compression enabled
- CDN integration for static assets

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL
psql -h localhost -U postgres -d prism_db

# Reset migrations
cd backend
npx prisma migrate reset
npx prisma db push
```

### Port Already in Use
```bash
# Backend
lsof -i :3001  # Find process, kill with: kill -9 <PID>

# Frontend
lsof -i :3000
```

### Correlation ID Tracking
All requests include `x-correlation-id` header for debugging:
```bash
curl -v http://localhost:3001/api/v1/dashboard/summary | grep correlation
```

## 📚 Documentation

- [API Documentation](./docs/API.md)
- [Database Schema](./docs/DATABASE.md)
- [Architecture Decision Records](./docs/ADR/)
- [Contributing Guide](./CONTRIBUTING.md)

## 🚦 CI/CD Pipeline

GitHub Actions workflow runs on every push:

1. **Unit Tests**: Backend with Jest, frontend with Vite
2. **Code Quality**: ESLint, TypeScript strict mode
3. **Security**: Trivy vulnerability scanning, npm audit
4. **Docker Build**: Multi-stage builds for optimization
5. **Deployment**: Automated deployment on merge to main

View pipeline status: `.github/workflows/ci-cd.yml`

## 📝 Logging & Debugging

### Log Levels
- `error`: Application errors
- `warn`: Warnings and degraded states
- `info`: General information
- `debug`: Detailed debugging info

### View Logs
```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Local logs
tail -f backend/logs/combined-*.log
tail -f backend/logs/error-*.log
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m "feat: description"`
4. Push to branch: `git push origin feature/name`
5. Open pull request

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@prism.local

## 🔮 Roadmap

- [ ] Authentication & Authorization (JWT/OAuth2)
- [ ] GraphQL API layer
- [ ] Advanced caching strategy
- [ ] Machine learning model versioning
- [ ] Multi-region deployment
- [ ] Advanced anomaly detection
- [ ] Custom alerting rules engine
- [ ] Integration with PagerDuty, Slack

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
