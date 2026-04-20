const serviceNames = [
  'api-gateway',
  'auth-service',
  'payment-service',
  'cart-service',
  'inventory-service',
  'notification-service',
  'order-service',
  'user-service',
  'product-service',
  'recommendation-service',
  'search-service',
  'cache-service',
  'db-proxy',
  'message-broker',
  'billing-service',
  'shipping-service',
  'analytics-service',
  'config-service',
  'logging-service',
  'trace-collector',
  'session-service',
  'fulfillment-service',
  'fraud-service',
  'catalog-service',
  'pricing-service',
  'review-service',
  'media-service',
  'checkout-service',
  'promotion-service',
  'loyalty-service',
  'edge-router',
  'scheduler-service',
  'report-service',
  'insight-service',
  'experiment-service',
  'audit-service',
  'notification-worker',
  'webhook-service',
  'warehouse-service',
  'catalog-indexer',
  'recommendation-worker',
  'billing-ledger-service',
  'telemetry-ingestor',
  'incident-service',
]

const dependencyMap = {
  'edge-router': ['api-gateway'],
  'api-gateway': ['auth-service', 'cart-service', 'catalog-service', 'order-service', 'search-service', 'recommendation-service'],
  'auth-service': ['user-service', 'session-service', 'audit-service'],
  'payment-service': ['billing-service', 'billing-ledger-service', 'fraud-service', 'db-proxy'],
  'cart-service': ['session-service', 'cache-service', 'catalog-service'],
  'inventory-service': ['warehouse-service', 'db-proxy', 'telemetry-ingestor'],
  'notification-service': ['notification-worker', 'webhook-service', 'message-broker'],
  'order-service': ['payment-service', 'inventory-service', 'shipping-service', 'notification-service', 'message-broker'],
  'user-service': ['session-service', 'audit-service', 'config-service'],
  'product-service': ['catalog-service', 'search-service', 'media-service'],
  'recommendation-service': ['analytics-service', 'experiment-service', 'catalog-indexer'],
  'search-service': ['catalog-indexer', 'cache-service'],
  'cache-service': ['db-proxy'],
  'db-proxy': ['incident-service'],
  'message-broker': ['logging-service'],
  'billing-service': ['billing-ledger-service', 'audit-service'],
  'shipping-service': ['warehouse-service', 'telemetry-ingestor'],
  'analytics-service': ['insight-service', 'report-service'],
  'config-service': ['logging-service'],
  'logging-service': ['trace-collector'],
  'trace-collector': ['insight-service'],
  'session-service': ['cache-service', 'auth-service'],
  'fulfillment-service': ['inventory-service', 'shipping-service'],
  'fraud-service': ['analytics-service', 'audit-service'],
  'catalog-service': ['catalog-indexer', 'media-service'],
  'pricing-service': ['product-service', 'analytics-service'],
  'review-service': ['catalog-service', 'analytics-service'],
  'media-service': ['trace-collector'],
  'checkout-service': ['cart-service', 'payment-service'],
  'promotion-service': ['pricing-service', 'loyalty-service'],
  'loyalty-service': ['user-service', 'analytics-service'],
  'scheduler-service': ['report-service', 'telemetry-ingestor'],
  'report-service': ['analytics-service'],
  'insight-service': ['audit-service'],
  'experiment-service': ['recommendation-service', 'analytics-service'],
  'audit-service': ['logging-service'],
  'notification-worker': ['message-broker', 'logging-service'],
  'webhook-service': ['api-gateway', 'message-broker'],
  'warehouse-service': ['db-proxy'],
  'catalog-indexer': ['search-service', 'logging-service'],
  'recommendation-worker': ['recommendation-service', 'analytics-service'],
  'billing-ledger-service': ['db-proxy', 'audit-service'],
  'telemetry-ingestor': ['trace-collector', 'analytics-service'],
  'incident-service': ['logging-service', 'message-broker'],
}

const timelineLabels = ['t1', 't2', 't3', 't4', 't5', 't6']

const serviceSummaries = {
  'api-gateway': 'Front-door traffic mesh with policy enforcement and request shaping.',
  'auth-service': 'Authentication and token exchange for internal and external callers.',
  'payment-service': 'Payment orchestration with retry logic and failure isolation.',
  'cart-service': 'Cart state and checkout session coordination.',
  'inventory-service': 'Inventory reservation and stock reconciliation.',
  'notification-service': 'Alert fan-out across email, push, and webhook channels.',
  'order-service': 'Order lifecycle manager with dependency fan-out.',
  'user-service': 'Identity profile, preferences, and user metadata.',
  'product-service': 'Product catalog enrichment and browsing context.',
  'recommendation-service': 'Ranking layer that scores the next best action.',
  'search-service': 'Query parsing, retrieval, and result ranking.',
  'cache-service': 'Latency shield for hot reads and repeated calls.',
  'db-proxy': 'Connection routing and pool protection for the datastore.',
  'message-broker': 'Event backbone for asynchronous fan-out.',
  'billing-service': 'Billing workflow, invoice generation, and reconciliation.',
  'shipping-service': 'Delivery orchestration and courier handoff.',
  'analytics-service': 'Usage telemetry aggregation and trend analysis.',
  'config-service': 'Runtime configuration distribution and rollout control.',
  'logging-service': 'Central log ingestion and retention layer.',
  'trace-collector': 'OpenTelemetry trace ingress and normalization.',
  'session-service': 'User session state and cache-backed token exchange.',
  'fulfillment-service': 'Warehouse fulfilment and packaging coordinator.',
  'fraud-service': 'Risk scoring and transaction anomaly detection.',
  'catalog-service': 'Product catalog serving and enrichment orchestration.',
  'pricing-service': 'Price rules, discounts, and margin guards.',
  'review-service': 'Ratings pipeline and sentiment enrichment.',
  'media-service': 'Asset delivery and media transformation.',
  'checkout-service': 'Checkout journey aggregator and state machine.',
  'promotion-service': 'Offer targeting and discount eligibility.',
  'loyalty-service': 'Points and reward redemption engine.',
  'edge-router': 'Ingress proxy and routing control point.',
  'scheduler-service': 'Cron-like background orchestration.',
  'report-service': 'Operational reporting and SLA summaries.',
  'insight-service': 'Cross-service anomaly synthesis and insights.',
  'experiment-service': 'Feature flag experiments and canary scoring.',
  'audit-service': 'Immutable event trail for compliance workflows.',
  'notification-worker': 'Background job execution for outbound alerts.',
  'webhook-service': 'External callback delivery with retry windows.',
  'warehouse-service': 'Physical inventory and warehouse coordination.',
  'catalog-indexer': 'Search indexing and catalog denormalization.',
  'recommendation-worker': 'Offline recommendation scoring jobs.',
  'billing-ledger-service': 'Ledger persistence and transactional bookkeeping.',
  'telemetry-ingestor': 'Batch telemetry ingress and normalization pipeline.',
  'incident-service': 'Incident triage and on-call workflow hub.',
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function hashSeed(input) {
  let value = 0

  for (let index = 0; index < input.length; index += 1) {
    value = (value * 31 + input.charCodeAt(index)) % 100000
  }

  return value
}

function buildScore(name, index, stepIndex) {
  return (hashSeed(name) + index * 17 + stepIndex * 29 + (index % 7) * 13) % 1000
}

function buildMetric(base, spread, stepIndex, index, severityBoost) {
  const wobble = ((index * 11 + stepIndex * 7) % spread) - spread / 2
  return clamp(base + wobble + severityBoost, 0, 100)
}

function createServiceSnapshots() {
  const serviceHistory = Object.fromEntries(serviceNames.map((name) => [name, []]))
  const timeline = []

  timelineLabels.forEach((label, stepIndex) => {
    const timestamp = new Date(Date.UTC(2026, 3, 20, 8 + stepIndex, stepIndex * 9, 0)).toISOString()
    const ranking = serviceNames
      .map((name, index) => ({ name, index, score: buildScore(name, index, stepIndex) }))
      .sort((left, right) => right.score - left.score)

    const critical = new Set(ranking.slice(0, 3).map((entry) => entry.name))
    const warning = new Set(ranking.slice(3, 9).map((entry) => entry.name))

    const snapshot = serviceNames.map((name, index) => {
      const severity = critical.has(name) ? 'critical' : warning.has(name) ? 'warning' : 'healthy'
      const severityBoost = severity === 'critical' ? 26 : severity === 'warning' ? 10 : -4
      const failureProbability = clamp(
        (buildScore(name, index, stepIndex) % 100) / 100 + (severity === 'critical' ? 0.22 : severity === 'warning' ? 0.08 : -0.1),
        0.03,
        0.98
      )
      const cpu = buildMetric(34 + (index % 7) * 4 + stepIndex * 1.5, 24, stepIndex, index, severityBoost)
      const memory = buildMetric(42 + (index % 5) * 5 + stepIndex, 20, stepIndex, index, severityBoost - 2)
      const latency = clamp(
        82 + (index % 9) * 14 + stepIndex * 12 + (severity === 'critical' ? 210 : severity === 'warning' ? 90 : -10),
        18,
        980
      )
      const errorRate = clamp(
        0.012 + ((index % 6) * 0.009) + (severity === 'critical' ? 0.135 : severity === 'warning' ? 0.045 : 0.004),
        0.002,
        0.68
      )

      const entry = {
        id: `${name}-${label}`,
        name,
        label,
        timestamp,
        status: severity,
        interaction_count: 90 + ((index * 37 + stepIndex * 17) % 410),
        metrics: {
          cpu: Number(cpu.toFixed(1)),
          memory: Number(memory.toFixed(1)),
          latency: Math.round(latency),
          error_rate: Number(errorRate.toFixed(3)),
          failure_probability: Number(failureProbability.toFixed(3)),
        },
        summary: serviceSummaries[name],
        upstream: dependencyMap[name] ? Object.entries(dependencyMap).filter(([, targets]) => targets.includes(name)).map(([source]) => source) : [],
        downstream: dependencyMap[name] || [],
      }

      serviceHistory[name].push(entry)
      return entry
    })

    timeline.push({
      label,
      timestamp,
      snapshot,
      failing: snapshot.filter((entry) => entry.status === 'critical').map((entry) => entry.name),
      atRisk: snapshot.filter((entry) => entry.status === 'warning').map((entry) => entry.name),
    })
  })

  const services = serviceNames.map((name, index) => {
    const latest = serviceHistory[name][serviceHistory[name].length - 1]
    const history = serviceHistory[name]
    const meanCpu = history.reduce((sum, entry) => sum + entry.metrics.cpu, 0) / history.length
    const meanLatency = history.reduce((sum, entry) => sum + entry.metrics.latency, 0) / history.length
    const failureProbability = history.reduce((sum, entry) => sum + entry.metrics.failure_probability, 0) / history.length

    return {
      id: name,
      name,
      description: serviceSummaries[name],
      status: latest.status,
      interaction_count: latest.interaction_count,
      failure_probability: latest.metrics.failure_probability,
      cpu: latest.metrics.cpu,
      memory: latest.metrics.memory,
      latency: latest.metrics.latency,
      error_rate: latest.metrics.error_rate,
      history,
      upstream: latest.upstream,
      downstream: latest.downstream,
      severityRank: latest.status === 'critical' ? 3 : latest.status === 'warning' ? 2 : 1,
      avgCpu: Number(meanCpu.toFixed(1)),
      avgLatency: Math.round(meanLatency),
      avgFailureProbability: Number(failureProbability.toFixed(3)),
      sortIndex: index,
    }
  })

  const alerts = timeline.flatMap((frame, frameIndex) =>
    frame.snapshot
      .filter((entry) => entry.status !== 'healthy')
      .sort((left, right) => right.metrics.failure_probability - left.metrics.failure_probability)
      .slice(0, 8)
      .map((entry, itemIndex) => ({
        id: `${frame.label}-${entry.name}`,
        service: entry.name,
        risk: entry.metrics.failure_probability,
        severity: entry.status === 'critical' ? 'high' : 'medium',
        timestamp: entry.timestamp,
        status: itemIndex === 0 && frameIndex % 2 === 0 ? 'open' : 'queued',
        acknowledged: itemIndex > 3,
        summary: `${entry.name} showed elevated ${entry.status === 'critical' ? 'failure' : 'degradation'} signals.`,
      }))
  ).slice(0, 24)

  const modelMetrics = {
    recall: 0.57,
    f1: 0.37,
    precision: 0.25,
    accuracy: 0.81,
    confusionMatrix: [
      [128, 34],
      [91, 44],
    ],
    lossCurve: [1.24, 1.08, 0.96, 0.84, 0.77, 0.71, 0.69, 0.64, 0.61, 0.58],
    prTradeoff: [
      { threshold: 0.15, precision: 0.17, recall: 0.86 },
      { threshold: 0.25, precision: 0.22, recall: 0.74 },
      { threshold: 0.35, precision: 0.25, recall: 0.57 },
      { threshold: 0.45, precision: 0.31, recall: 0.46 },
      { threshold: 0.55, precision: 0.39, recall: 0.34 },
      { threshold: 0.65, precision: 0.47, recall: 0.22 },
    ],
  }

  const dashboardTrend = timeline.map((frame, index) => ({
    label: frame.label,
    failureProbability: Number((frame.snapshot.filter((entry) => entry.status === 'critical').length / frame.snapshot.length * 100).toFixed(1)),
    latency: Math.round(frame.snapshot.reduce((sum, entry) => sum + entry.metrics.latency, 0) / frame.snapshot.length),
    risk: Number((frame.snapshot.reduce((sum, entry) => sum + entry.metrics.failure_probability, 0) / frame.snapshot.length * 100).toFixed(1)),
    index,
  }))

  const serviceSeries = Object.fromEntries(
    serviceNames.map((name) => [
      name,
      serviceHistory[name].map((entry, index) => ({
        label: timelineLabels[index],
        timestamp: entry.timestamp,
        cpu: entry.metrics.cpu,
        memory: entry.metrics.memory,
        latency: entry.metrics.latency,
        errorRate: Number((entry.metrics.error_rate * 100).toFixed(2)),
        failureProbability: Number((entry.metrics.failure_probability * 100).toFixed(1)),
      })),
    ])
  )

  return {
    serviceNames,
    dependencyMap,
    timeline,
    services,
    alerts,
    modelMetrics,
    dashboardTrend,
    serviceSeries,
    latestSnapshot: timeline[timeline.length - 1].snapshot,
  }
}

export const mockOpsData = createServiceSnapshots()

export function getServiceById(id) {
  return mockOpsData.services.find((service) => service.id === id)
}

export function getStatusLabel(status) {
  if (status === 'critical') {
    return 'FAILURE PREDICTED'
  }

  if (status === 'warning') {
    return 'AT RISK'
  }

  return 'NORMAL'
}
