#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimulation");

namespace
{
  constexpr uint32_t kNumSta = 30;         // number of stations
  constexpr double kSimulationTime = 10.0; // total simulation time (s)
  constexpr double kEnvStepTime = 0.01;     // gym step interval (s)
  constexpr uint16_t kOpenGymPort = 5556;  // OpenGym server port
  constexpr uint32_t kActionCount = 7;     // number of discrete actions
  constexpr uint32_t kDefaultCwMin = 7;    // default CW Min of stations
  constexpr uint32_t kDefaultCwMax = 1023; // default CW Max of stations
}

// Variables for runtime statistics
static double s_totalBytesReceived = 0;
static uint32_t s_collisionCount = 0;
static uint32_t s_totalTxCount = 0;
static double s_episodeSimulationTime = 0;
static double s_envStemTime = 5;

// per-node counters:
static std::vector<uint32_t> s_collisionCountVec;
static std::vector<uint32_t> s_totalTxCountVec;

const std::vector<std::string> features = {"N_STAs", "Throughput", "Collision_probability"};

Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  float low = 0.0;
  float high = 100.0;
  std::vector<uint32_t> shape = {
      nodeNum,
  };
  std::string dtype = TypeNameGet<uint32_t>();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
  NS_LOG_UNCOND("MyGetObservationSpace: " << space);
  return space;
}

Ptr<OpenGymSpace> MyGetActionSpace(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  float low = 0.0;
  float high = 100.0;
  std::vector<uint32_t> shape = {
      nodeNum, 2};
  std::string dtype = TypeNameGet<uint32_t>();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
  NS_LOG_UNCOND("MyGetActionSpace: " << space);
  return space;
}

Ptr<WifiMacQueue> GetQueue(Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice(0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
  PointerValue ptr;

  wifi_mac->GetAttribute("BE_Txop", ptr);
  Ptr<Txop> txop = ptr.Get<Txop>();
  Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue();
  return queue;
}

Ptr<OpenGymDataContainer> MyGetObservation(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  std::vector<uint32_t> shape = {
      nodeNum,
      2};
  Ptr<OpenGymBoxContainer<uint32_t>> box = CreateObject<OpenGymBoxContainer<uint32_t>>(shape);

  size_t idx = 0;
  for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i, ++idx)
  {
    Ptr<Node> node = *i;
    Ptr<WifiMacQueue> queue = GetQueue(node);
    uint32_t nPackets = queue->GetNPackets();

    box->AddValue(nPackets);

    uint8_t collisionProbability = 0;
    // Calculating collision probability. TODO: add explanation on the calc process
    if (s_totalTxCountVec[idx] > 0)
    {
      double ratio = double(s_collisionCountVec[idx]) / double(s_totalTxCountVec[idx]);
      // ceil(ratio * 255) produces a double in [0.0 … 255.0]
      collisionProbability = static_cast<uint8_t>(std::ceil(ratio * 255.0));
    }

    box->AddValue(collisionProbability);
  }

  NS_LOG_DEBUG("MyGetObservation: " << box);
  return box;
}

float MyGetReward(void)
{
  const double throughput = s_totalBytesReceived * 8.0 / 1e6 / s_envStemTime;

  std::cout << Simulator::Now().GetSeconds()
            << "s - Throughput: " << throughput
            << " Mbps, Collisions: " << s_collisionCount
            << std::endl;

  s_collisionCount = 0;
  s_totalTxCount = 0;
  s_totalBytesReceived = 0;

  std::fill(s_collisionCountVec.begin(),
            s_collisionCountVec.end(), 0);
  std::fill(s_totalTxCountVec.begin(),
            s_totalTxCountVec.end(), 0);

  return static_cast<float>(throughput);
}

static void PhyRxDrop(std::string context,
                      Ptr<const Packet> packet, const ns3::WifiPhyRxfailureReason reason)
{
  // context will be the trace‐source path, e.g.
  // "/NodeList/0/DeviceList/0/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/PhyRxDrop"

  auto a = context.find("/NodeList/") + 10;
  auto b = context.find("/DeviceList", a);
  uint32_t nodeId = std::stoul(context.substr(a, b - a));

  switch (reason)
  {
  case ns3::WifiPhyRxfailureReason::PREAMBLE_DETECT_FAILURE:
  case ns3::WifiPhyRxfailureReason::PREAMBLE_DETECTION_PACKET_SWITCH:
  case ns3::WifiPhyRxfailureReason::FRAME_CAPTURE_PACKET_SWITCH:
  {
    ++s_collisionCount;
    ++s_collisionCountVec[nodeId];
    break;
  }
  default:
    break;
  }
}

static void PhyTxBegin(std::string context, Ptr<const Packet> packet, double txPowerDbm)
{
  s_totalTxCount++;

  auto a = context.find("/NodeList/") + 10;
  auto b = context.find("/DeviceList", a);
  uint32_t nodeId = std::stoul(context.substr(a, b - a));
  ++s_totalTxCountVec[nodeId];
}

void CalculateThroughput()
{
  // std::cout << Simulator::Now().GetSeconds() << "s - Throughput: " << totalThroughput << " Mbps, Collisions: " << s_collisionCount << std::endl;
  // totalThroughput = 0; // Reset for next interval
  // Simulator::Schedule(Seconds(1.0), &CalculateThroughput);
}

bool SetCw(Ptr<Node> node, uint32_t cwMinValue = 0, uint32_t cwMaxValue = 0)
{
  Ptr<NetDevice> dev = node->GetDevice(0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
  NS_ASSERT(wifi_mac != nullptr);
  PointerValue ptr;
  if (!wifi_mac->GetAttributeFailSafe("BE_Txop", ptr))
  {
    NS_LOG_UNCOND("Failed to get Txop");
    return false;
  }
  Ptr<Txop> txop = ptr.Get<Txop>();
  NS_ASSERT(txop != nullptr);

  // if both set to the same value then we have uniform backoff?
  if (cwMinValue != 0)
  {
    NS_LOG_DEBUG("Set CW min: " << cwMinValue);
    txop->SetMinCw(cwMinValue);
  }

  if (cwMaxValue != 0)
  {
    NS_LOG_DEBUG("Set CW max: " << cwMaxValue);
    txop->SetMaxCw(cwMaxValue);
  }
  return true;
}

bool SetCwForAllNodes(uint32_t cwMinValue = 0, uint32_t cwMaxValue = 0)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes(); ++i)
  {
    SetCw(NodeList::GetNode(i), cwMinValue, kDefaultCwMax);
  }
  return true;
}

bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_UNCOND("MyExecuteActions: " << action);

  Ptr<OpenGymBoxContainer<uint32_t>> box = DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
  std::vector<uint32_t> actionVector = box->GetData();

  uint32_t nodeNum = NodeList::GetNNodes();
  for (uint32_t i = 0; i < nodeNum; i++)
  {
    Ptr<Node> node = NodeList::GetNode(i);
    uint32_t cwSize = actionVector.at(i);
    // TODO: shouuld we limit to 7, 15, 31, 63, etc. ?
    SetCw(node, cwSize, cwSize);
  }

  return true;
}

void PhyRxOk(std::string context,
             Ptr<const Packet> packet,
             double snr,
             WifiMode mode,
             WifiPreamble preamble)
{
  s_totalBytesReceived += packet->GetSize();
}

bool MyGetGameOver(void)
{
  NS_LOG_DEBUG("Sim Time: " << Simulator::Now().GetSeconds());
  return (Simulator::Now().GetSeconds() >= s_episodeSimulationTime);
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
  Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
  openGymInterface->NotifyCurrentState();
}
int main(int argc, char *argv[])
{
  uint32_t nSta = kNumSta;
  double simulationTime = kSimulationTime;
  double envStepTime = kEnvStepTime;
  uint16_t openGymPort = kOpenGymPort;
  uint64_t simSeed = 0;
  bool startSim = true;
  bool debug = false;

  CommandLine cmd;
  cmd.AddValue("openGymPort", "OpenGym server port", openGymPort);
  cmd.AddValue("stepTime", "Env step interval (s)", envStepTime);
  cmd.AddValue("simTime", "Total simulation time (s)", simulationTime);
  cmd.AddValue("simSeed", "RNG seed", simSeed);
  cmd.AddValue("startSim", "Whether to start simulation immediately", startSim);
  cmd.AddValue("debug", "Enable debug logging", debug);
  cmd.AddValue("nodeNum", "Number of stations", nSta);
  cmd.Parse(argc, argv);

  s_episodeSimulationTime = simulationTime;
  s_envStemTime = envStepTime;

  NodeContainer wifiStaNodes, wifiApNode;
  wifiStaNodes.Create(nSta);
  wifiApNode.Create(1);

  s_collisionCountVec.assign(nSta, 0);
  s_totalTxCountVec.assign(nSta, 0);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211n);

  WifiMacHelper mac;
  Ssid ssid = Ssid("ns3-wifi");

  mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false), "QosSupported", BooleanValue(true));
  NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

  mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "QosSupported", BooleanValue(true));
  NetDeviceContainer apDevice = wifi.Install(phy, mac, wifiApNode);

  MobilityHelper mobility;
  mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                "MinX", DoubleValue(0.0),
                                "MinY", DoubleValue(0.0),
                                "DeltaX", DoubleValue(5.0),
                                "DeltaY", DoubleValue(10.0),
                                "GridWidth", UintegerValue(3),
                                "LayoutType", StringValue("RowFirst"));
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(wifiStaNodes);
  mobility.Install(wifiApNode);

  InternetStackHelper stack;
  stack.Install(wifiStaNodes);
  stack.Install(wifiApNode);

  Ipv4AddressHelper address;
  address.SetBase("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);
  Ipv4InterfaceContainer apInterface = address.Assign(apDevice);

  UdpServerHelper server(9);
  ApplicationContainer serverApp = server.Install(wifiApNode.Get(0));
  serverApp.Start(Seconds(1.0));
  serverApp.Stop(Seconds(simulationTime));

  UdpClientHelper client(apInterface.GetAddress(0), 9);
  client.SetAttribute("MaxPackets", UintegerValue(100000));
  client.SetAttribute("Interval", TimeValue(MilliSeconds(1)));
  client.SetAttribute("PacketSize", UintegerValue(1024));

  ApplicationContainer clientApps;
  for (uint32_t i = 0; i < nSta; ++i)
  {
    clientApps.Add(client.Install(wifiStaNodes.Get(i)));
  }
  clientApps.Start(Seconds(2.0));
  clientApps.Stop(Seconds(simulationTime));

  Config::Connect("/NodeList/*/DeviceList/*/Phy/State/RxOk",
                  MakeCallback(&PhyRxOk));
  Config::Connect(
      "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/PhyRxDrop",
      MakeCallback(&PhyRxDrop));
  Config::Connect(
      "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/PhyTxBegin",
      MakeCallback(&PhyTxBegin));

  SetCwForAllNodes(kDefaultCwMin, kDefaultCwMax);

  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
  openGymInterface->SetGetActionSpaceCb(MakeCallback(&MyGetActionSpace));
  openGymInterface->SetGetObservationSpaceCb(MakeCallback(&MyGetObservationSpace));
  openGymInterface->SetGetGameOverCb(MakeCallback(&MyGetGameOver));
  openGymInterface->SetGetObservationCb(MakeCallback(&MyGetObservation));
  openGymInterface->SetGetRewardCb(MakeCallback(&MyGetReward));
  openGymInterface->SetExecuteActionsCb(MakeCallback(&MyExecuteActions));
  Simulator::Schedule(Seconds(0.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
  Simulator::Stop(Seconds(simulationTime + envStepTime));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}
