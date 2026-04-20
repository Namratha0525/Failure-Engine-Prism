import { useEffect, useRef } from 'react'
import { createRealtimeClient } from '../services/realtime'
import { useDashboardStore } from '../store/useDashboardStore'

export function useDashboardBootstrap() {
  const bootstrapDashboard = useDashboardStore((state) => state.bootstrapDashboard)
  const refreshDashboard = useDashboardStore((state) => state.refreshDashboard)
  const refreshServiceDetail = useDashboardStore((state) => state.refreshServiceDetail)
  const recordSocketEvent = useDashboardStore((state) => state.recordSocketEvent)
  const setConnectionStatus = useDashboardStore((state) => state.setConnectionStatus)
  const selectedServiceName = useDashboardStore((state) => state.selectedServiceName)

  const socketRef = useRef(null)
  const refreshTimerRef = useRef(null)
  const selectedServiceRef = useRef(selectedServiceName)

  useEffect(() => {
    selectedServiceRef.current = selectedServiceName
  }, [selectedServiceName])

  useEffect(() => {
    let mounted = true

    bootstrapDashboard().catch(() => {
      if (mounted) {
        setConnectionStatus('degraded')
      }
    })

    const socket = createRealtimeClient()
    socketRef.current = socket

    const queueRefresh = () => {
      if (refreshTimerRef.current) {
        window.clearTimeout(refreshTimerRef.current)
      }

      refreshTimerRef.current = window.setTimeout(() => {
        refreshDashboard()

        if (selectedServiceRef.current) {
          refreshServiceDetail(selectedServiceRef.current)
        }
      }, 450)
    }

    socket.on('connect', () => setConnectionStatus('live'))
    socket.on('disconnect', () => setConnectionStatus('offline'))
    socket.on('connect_error', () => setConnectionStatus('reconnecting'))

    socket.on('new_telemetry', (payload) => {
      recordSocketEvent('new_telemetry', payload)
      queueRefresh()
    })

    socket.on('new_prediction', (payload) => {
      recordSocketEvent('new_prediction', payload)
      queueRefresh()
    })

    socket.on('service_alert', (payload) => {
      recordSocketEvent('service_alert', payload)
      queueRefresh()
    })

    socket.on('dashboard_update', (payload) => {
      recordSocketEvent('dashboard_update', payload)
      queueRefresh()
    })

    return () => {
      mounted = false

      if (refreshTimerRef.current) {
        window.clearTimeout(refreshTimerRef.current)
      }

      if (socketRef.current) {
        socketRef.current.removeAllListeners()
        socketRef.current.disconnect()
      }

      setConnectionStatus('connecting')
    }
  }, [bootstrapDashboard, recordSocketEvent, refreshDashboard, refreshServiceDetail, setConnectionStatus])
}